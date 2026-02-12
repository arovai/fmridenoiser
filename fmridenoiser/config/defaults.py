"""Default configuration dataclasses for fMRI Denoiser."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class MotionCensoringConfig:
    """Configuration for motion-based censoring (FD-based scrubbing).
    
    Attributes:
        enabled: Whether motion censoring is enabled.
        fd_threshold: Framewise displacement threshold in cm (fMRIPrep reports FD in cm).
        fd_column: Column name for FD in confounds file.
        extend_before: Also censor N volumes before high-motion.
        extend_after: Also censor N volumes after high-motion.
        min_segment_length: Minimum contiguous segment length to keep (scrubbing).
    """
    enabled: bool = False
    fd_threshold: float = 0.5
    fd_column: str = "framewise_displacement"
    extend_before: int = 0
    extend_after: int = 0
    min_segment_length: int = 0


@dataclass
class CensoringConfig:
    """Configuration for temporal censoring in denoising.
    
    This covers FD-based motion censoring and dummy scan removal,
    which are preprocessing concerns. Condition-based masking is
    NOT handled here — it belongs in downstream analysis tools.
    
    Attributes:
        enabled: Master switch for censoring.
        drop_initial_volumes: Number of initial volumes to drop (dummy scans).
        motion_censoring: FD-based motion censoring configuration.
        custom_mask_file: Path to custom censoring mask TSV.
        min_volumes_retained: Minimum number of volumes required after censoring.
        min_fraction_retained: Minimum fraction of volumes required.
        warn_fraction_retained: Warn if retention falls below this.
    """
    enabled: bool = False
    drop_initial_volumes: int = 0
    motion_censoring: MotionCensoringConfig = field(default_factory=MotionCensoringConfig)
    custom_mask_file: Optional[Path] = None
    min_volumes_retained: int = 50
    min_fraction_retained: float = 0.3
    warn_fraction_retained: float = 0.5


@dataclass
class DenoisingConfig:
    """Configuration for the fMRI denoising pipeline.
    
    Attributes:
        subject: List of subject IDs to process
        tasks: List of task names to process
        sessions: List of session IDs to process
        runs: List of run IDs to process
        spaces: List of space names to process
        label: Custom label for output filenames
        denoising_strategy: Name of predefined denoising strategy (if used)
        confounds: List of confound column names for regression
        high_pass: High-pass filter cutoff in Hz
        low_pass: Low-pass filter cutoff in Hz
        ica_aroma: Use ICA-AROMA denoised files
        reference_functional_file: Reference image path or "first_functional_file"
        overwrite: Whether to re-denoise if files exist
        censoring: Censoring configuration (FD-based + dummy scan removal)
    """
    
    # BIDS entity filters
    subject: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    sessions: Optional[List[str]] = None
    runs: Optional[List[str]] = None
    spaces: Optional[List[str]] = None
    
    # Custom label for output filenames
    label: Optional[str] = None
    
    # Denoising parameters
    denoising_strategy: Optional[str] = None
    confounds: List[str] = field(default_factory=lambda: [
        "csf", "white_matter",
        "trans_x", "trans_y", "trans_z",
        "rot_x", "rot_y", "rot_z"
    ])
    high_pass: float = 0.01
    low_pass: float = 0.08
    ica_aroma: bool = False
    reference_functional_file: str = "first_functional_file"
    overwrite: bool = False
    
    # Censoring configuration
    censoring: CensoringConfig = field(default_factory=CensoringConfig)
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        from fmridenoiser.config.validator import ConfigValidator
        
        validator = ConfigValidator()
        
        # Validate filter values
        validator.validate_alpha(self.high_pass, "high_pass")
        validator.validate_alpha(self.low_pass, "low_pass")
        
        # Validate ICA-AROMA incompatibility with motion parameters
        if self.ica_aroma:
            motion_params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
            motion_in_confounds = any(mp in self.confounds for mp in motion_params)
            if motion_in_confounds:
                validator.errors.append(
                    "ICA-AROMA is incompatible with motion parameters in confounds. "
                    "Remove motion parameters from confounds list."
                )
        
        validator.raise_if_errors()
