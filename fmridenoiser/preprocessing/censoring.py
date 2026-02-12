"""FD-based temporal censoring for fMRI time series.

Temporal censoring removes specific timepoints (volumes) from fMRI data.
This module handles:

1. **Dummy scan removal**: Discard initial volumes during scanner equilibration.
2. **Motion scrubbing**: Remove timepoints with excessive head motion (FD-based).
3. **Segment filtering**: Remove short contiguous segments after scrubbing.
4. **Custom censoring**: Apply user-defined censoring masks.

Note: Condition-based temporal selection ("condition masking") is handled
separately in connectomix, not in fmridenoiser.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import nibabel as nib
import numpy as np
import pandas as pd

from fmridenoiser.utils.exceptions import PreprocessingError


logger = logging.getLogger(__name__)


class TemporalCensor:
    """Generate and apply FD-based temporal censoring masks.
    
    This class manages the creation and application of temporal masks
    for censoring fMRI volumes based on motion (framewise displacement).
    
    Attributes:
        config: Censoring configuration from DenoisingConfig.
        n_volumes: Number of volumes in the original data.
        tr: Repetition time in seconds.
        mask: Boolean mask (True = keep, False = censor).
        censoring_log: Dictionary tracking censoring reasons per volume.
    """
    
    def __init__(
        self,
        config,
        n_volumes: int,
        tr: float,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize temporal censor.
        
        Args:
            config: CensoringConfig from fmridenoiser.config.defaults.
            n_volumes: Number of volumes in the data.
            tr: Repetition time in seconds.
            logger: Optional logger instance.
        """
        self.config = config
        self.n_volumes = n_volumes
        self.tr = tr
        self._logger = logger or logging.getLogger(__name__)
        
        self.mask = np.ones(n_volumes, dtype=bool)
        self.censoring_log: Dict[int, List[str]] = {}
        
        self._logger.debug(f"Initialized temporal censor for {n_volumes} volumes (TR={tr}s)")
    
    def apply_initial_drop(self) -> int:
        """Mark first N volumes for censoring.
        
        Returns:
            Number of volumes marked for censoring.
        """
        n_drop = self.config.drop_initial_volumes
        if n_drop <= 0:
            return 0
        
        if n_drop >= self.n_volumes:
            raise PreprocessingError(
                f"Cannot drop {n_drop} initial volumes from data with only "
                f"{self.n_volumes} volumes."
            )
        
        for i in range(n_drop):
            self.mask[i] = False
            self._add_censoring_reason(i, "initial_drop")
        
        self._logger.info(f"Temporal censoring: marked {n_drop} initial volumes for removal")
        return n_drop
    
    def apply_motion_censoring(self, confounds_df: pd.DataFrame) -> int:
        """Mark high-motion volumes for censoring based on FD.
        
        Args:
            confounds_df: Confounds DataFrame with FD column.
            
        Returns:
            Number of volumes marked for censoring.
        """
        mc = self.config.motion_censoring
        if not mc.enabled:
            return 0
        
        if mc.fd_column not in confounds_df.columns:
            available = [c for c in confounds_df.columns if 'displacement' in c.lower() or 'fd' in c.lower()]
            raise PreprocessingError(
                f"FD column '{mc.fd_column}' not found in confounds. "
                f"Available motion-related columns: {available}"
            )
        
        fd_values = confounds_df[mc.fd_column].values
        fd_values = np.nan_to_num(fd_values, nan=0.0)
        
        high_motion = fd_values > mc.fd_threshold
        
        if mc.extend_before > 0 or mc.extend_after > 0:
            extended = np.zeros_like(high_motion)
            for i, is_high in enumerate(high_motion):
                if is_high:
                    start = max(0, i - mc.extend_before)
                    end = min(self.n_volumes, i + mc.extend_after + 1)
                    extended[start:end] = True
            high_motion = extended
        
        n_censored = 0
        for i, censor in enumerate(high_motion):
            if censor:
                if self.mask[i]:
                    n_censored += 1
                self.mask[i] = False
                self._add_censoring_reason(i, f"motion_fd>{mc.fd_threshold}cm")
        
        try:
            mm_equiv = float(mc.fd_threshold) * 10.0
            self._logger.info(
                f"Temporal censoring: marked {n_censored} volumes for motion "
                f"(FD > {mc.fd_threshold} cm ({mm_equiv:.2f} mm))"
            )
        except Exception:
            self._logger.info(
                f"Temporal censoring: marked {n_censored} volumes for motion "
                f"(FD > {mc.fd_threshold} cm)"
            )
        return n_censored
    
    def apply_segment_filtering(self, min_segment_length: int) -> int:
        """Remove continuous segments shorter than min_segment_length.
        
        After motion censoring, identifies contiguous runs of kept volumes
        and censors segments shorter than the specified minimum length.
        
        Args:
            min_segment_length: Minimum number of contiguous volumes required.
        
        Returns:
            Number of additional volumes censored.
        """
        if min_segment_length <= 0:
            return 0
        
        n_censored = 0
        segment_start = None
        segments = []
        
        for i in range(self.n_volumes):
            if self.mask[i]:
                if segment_start is None:
                    segment_start = i
            else:
                if segment_start is not None:
                    segments.append((segment_start, i))
                    segment_start = None
        
        if segment_start is not None:
            segments.append((segment_start, self.n_volumes))
        
        for start, end in segments:
            segment_length = end - start
            if segment_length < min_segment_length:
                for i in range(start, end):
                    self.mask[i] = False
                    self._add_censoring_reason(i, f"segment_too_short<{min_segment_length}")
                    n_censored += 1
        
        if n_censored > 0:
            self._logger.info(
                f"Temporal censoring: marked {n_censored} additional volumes "
                f"(segments shorter than {min_segment_length} volumes)"
            )
        
        return n_censored
    
    def apply_custom_mask(self, mask_file: Path) -> int:
        """Apply user-provided censoring mask.
        
        Args:
            mask_file: Path to TSV file with 'censor' column (1=keep, 0=drop).
            
        Returns:
            Number of volumes marked for censoring.
        """
        if mask_file is None:
            return 0
        
        mask_file = Path(mask_file)
        if not mask_file.exists():
            raise PreprocessingError(f"Custom mask file not found: {mask_file}")
        
        mask_df = pd.read_csv(mask_file, sep='\t')
        
        if 'censor' not in mask_df.columns:
            raise PreprocessingError(
                f"Custom mask file must have 'censor' column. "
                f"Found columns: {list(mask_df.columns)}"
            )
        
        custom_mask = mask_df['censor'].values.astype(bool)
        
        if len(custom_mask) != self.n_volumes:
            raise PreprocessingError(
                f"Custom mask length ({len(custom_mask)}) doesn't match "
                f"number of volumes ({self.n_volumes})"
            )
        
        n_censored = 0
        for i, keep in enumerate(custom_mask):
            if not keep:
                if self.mask[i]:
                    n_censored += 1
                self.mask[i] = False
                self._add_censoring_reason(i, "custom_mask")
        
        self._logger.info(f"Temporal censoring: marked {n_censored} volumes from custom mask")
        return n_censored
    
    def _add_censoring_reason(self, volume_idx: int, reason: str) -> None:
        """Record reason for censoring a volume."""
        if volume_idx not in self.censoring_log:
            self.censoring_log[volume_idx] = []
        if reason not in self.censoring_log[volume_idx]:
            self.censoring_log[volume_idx].append(reason)
    
    def get_mask(self) -> np.ndarray:
        """Return the combined boolean mask (True = keep, False = censor)."""
        return self.mask.copy()
    
    def validate(self) -> None:
        """Check if enough volumes remain after censoring.
        
        Raises:
            PreprocessingError: If too few volumes remain.
        """
        self._warnings: List[str] = []
        
        n_retained = np.sum(self.mask)
        fraction_retained = n_retained / self.n_volumes
        
        if n_retained < self.config.min_volumes_retained:
            warning_msg = (
                f"LOW VOLUME COUNT after censoring: only {n_retained} volumes remaining "
                f"(recommended minimum: {self.config.min_volumes_retained}). "
                f"Results may be unreliable."
            )
            self._warnings.append(warning_msg)
            self._logger.warning(warning_msg)
        
        if fraction_retained < self.config.min_fraction_retained:
            warning_msg = (
                f"LOW RETENTION RATE after censoring: {fraction_retained:.1%} remaining "
                f"(recommended minimum: {self.config.min_fraction_retained:.0%}). "
                f"Results may be unreliable."
            )
            self._warnings.append(warning_msg)
            self._logger.warning(warning_msg)
        elif fraction_retained < self.config.warn_fraction_retained:
            self._logger.warning(
                f"Only {fraction_retained:.1%} of volumes retained after censoring. "
                f"Interpret results with caution."
            )
    
    def apply_to_image(
        self,
        img: nib.Nifti1Image,
    ) -> nib.Nifti1Image:
        """Apply censoring mask to 4D image.
        
        Args:
            img: 4D NIfTI image.
            
        Returns:
            New image with censored volumes removed.
        """
        data = img.get_fdata()
        
        if data.ndim != 4:
            raise PreprocessingError(f"Expected 4D image, got {data.ndim}D")
        
        censored_data = data[..., self.mask]
        new_img = nib.Nifti1Image(censored_data, img.affine, img.header)
        new_img.header.set_data_shape(censored_data.shape)
        
        n_original = data.shape[-1]
        n_retained = censored_data.shape[-1]
        self._logger.debug(f"Applied censoring: {n_original} -> {n_retained} volumes")
        
        return new_img
    
    def apply_to_confounds(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply censoring mask to confounds DataFrame.
        
        Args:
            df: Confounds DataFrame.
            
        Returns:
            New DataFrame with censored rows removed.
        """
        return df.iloc[self.mask].reset_index(drop=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Return censoring statistics for reporting."""
        n_retained = int(np.sum(self.mask))
        n_censored = self.n_volumes - n_retained
        
        reason_counts = {}
        for reasons in self.censoring_log.values():
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        summary = {
            'enabled': self.config.enabled,
            'n_original': self.n_volumes,
            'n_retained': n_retained,
            'n_censored': n_censored,
            'fraction_retained': float(n_retained / self.n_volumes),
            'reason_counts': reason_counts,
            'mask': self.mask.tolist(),
        }
        
        if hasattr(self, '_warnings') and self._warnings:
            summary['warnings'] = self._warnings
        
        return summary
    
    def get_censoring_entity(self) -> Optional[str]:
        """Generate BIDS-style entity string for censoring.
        
        Returns:
            Entity string like "drop4fd05" or None if no censoring.
        """
        if not self.config.enabled:
            return None
        
        parts = []
        
        if self.config.drop_initial_volumes > 0:
            parts.append(f"drop{self.config.drop_initial_volumes}")
        
        if self.config.motion_censoring.enabled:
            fd_str = str(self.config.motion_censoring.fd_threshold).replace('.', '')
            parts.append(f"fd{fd_str}")
        
        if self.config.custom_mask_file:
            parts.append("custom")
        
        if not parts:
            return None
        
        return "".join(parts)
