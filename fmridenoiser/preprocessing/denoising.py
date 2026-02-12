"""Functional image denoising."""

from pathlib import Path
from typing import List, Optional
import logging
import json
import nibabel as nib
from nilearn import image
import numpy as np

from fmridenoiser.io.readers import load_confounds, get_repetition_time
from fmridenoiser.utils.exceptions import PreprocessingError


def denoise_image(
    img_path: Path,
    confounds_path: Path,
    confound_names: List[str],
    high_pass: float,
    low_pass: float,
    output_path: Path,
    logger: logging.Logger,
    t_r: Optional[float] = None,
    overwrite: bool = True
) -> Path:
    """Denoise functional image using confound regression and filtering.
    
    Args:
        img_path: Path to functional image
        confounds_path: Path to confounds TSV file
        confound_names: List of confound column names to regress out
        high_pass: High-pass filter cutoff in Hz
        low_pass: Low-pass filter cutoff in Hz
        output_path: Path for denoised output
        logger: Logger instance
        t_r: Repetition time in seconds (will be read from JSON if None)
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to denoised image
    """
    if output_path.exists() and not overwrite:
        _validate_existing_denoised_file(
            output_path=output_path,
            confound_names=confound_names,
            high_pass=high_pass,
            low_pass=low_pass,
            logger=logger,
        )
        logger.info(f"Denoised file exists with matching parameters, skipping: {output_path.name}")
        return output_path
    
    logger.info(f"Denoising {img_path.name}")
    
    try:
        confounds, expanded_names = load_confounds(confounds_path, confound_names)
        logger.debug(f"  Loaded {len(expanded_names)} confound(s): {expanded_names[:5]}{'...' if len(expanded_names) > 5 else ''}")
        
        if t_r is None:
            json_path = img_path.with_suffix('').with_suffix('.json')
            if json_path.exists():
                t_r = get_repetition_time(json_path)
                logger.debug(f"  TR = {t_r}s")
            else:
                img = nib.load(img_path)
                if len(img.header.get_zooms()) > 3:
                    t_r = float(img.header.get_zooms()[3])
                    logger.debug(f"  TR from header = {t_r}s")
                else:
                    logger.warning("  No JSON sidecar and TR not in NIfTI header - filtering will not work correctly")
        
        img = nib.load(img_path)
        cleaned = image.clean_img(
            img,
            confounds=confounds,
            high_pass=high_pass,
            low_pass=low_pass,
            standardize=True,
            detrend=True,
            t_r=t_r
        )
        
        logger.debug(f"  Applied: high_pass={high_pass}Hz, low_pass={low_pass}Hz, standardize=True, detrend=True")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_filename(str(output_path))
        
        sidecar_data = {
            'Description': 'Denoised fMRI data produced by fmridenoiser',
            'Confounds': confound_names,
            'ExpandedConfounds': expanded_names,
            'HighPass_Hz': high_pass,
            'LowPass_Hz': low_pass,
            'Standardized': True,
            'Detrended': True,
            'RepetitionTime': t_r
        }
        
        sidecar_path = output_path.with_suffix('').with_suffix('.json')
        with sidecar_path.open('w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        logger.debug(f"  Saved to: {output_path}")
        return output_path
    
    except Exception as e:
        raise PreprocessingError(f"Failed to denoise {img_path.name}: {e}")


def compute_denoising_quality_metrics(
    original_img: nib.Nifti1Image,
    denoised_img: nib.Nifti1Image,
    mask_img: Optional[nib.Nifti1Image] = None
) -> dict:
    """Compute quality metrics for denoising.
    
    Returns:
        Dictionary with quality metrics (tSNR, std reduction, etc.)
    """
    original_data = original_img.get_fdata()
    denoised_data = denoised_img.get_fdata()
    
    if mask_img is not None:
        mask_data = mask_img.get_fdata().astype(bool)
        original_masked = original_data[mask_data]
        denoised_masked = denoised_data[mask_data]
    else:
        original_masked = original_data.reshape(-1, original_data.shape[-1])
        denoised_masked = denoised_data.reshape(-1, denoised_data.shape[-1])
    
    metrics = {}
    metrics['temporal_std_original'] = float(np.nanmean(np.nanstd(original_masked, axis=-1)))
    metrics['temporal_std_denoised'] = float(np.nanmean(np.nanstd(denoised_masked, axis=-1)))
    metrics['std_reduction_percent'] = float(
        100 * (1 - metrics['temporal_std_denoised'] / metrics['temporal_std_original'])
    )
    
    tsnr_original = np.nanmean(original_masked, axis=-1) / (np.nanstd(original_masked, axis=-1) + 1e-10)
    tsnr_denoised = np.nanmean(denoised_masked, axis=-1) / (np.nanstd(denoised_masked, axis=-1) + 1e-10)
    
    metrics['tsnr_original'] = float(np.nanmean(tsnr_original))
    metrics['tsnr_denoised'] = float(np.nanmean(tsnr_denoised))
    metrics['tsnr_improvement_percent'] = float(
        100 * (metrics['tsnr_denoised'] - metrics['tsnr_original']) / (metrics['tsnr_original'] + 1e-10)
    )
    
    return metrics


def compute_denoising_histogram_data(
    original_img: nib.Nifti1Image,
    denoised_img: nib.Nifti1Image,
    mask_img: Optional[nib.Nifti1Image] = None,
    n_bins: int = 100,
    subsample: int = 10
) -> dict:
    """Compute histogram data for before/after denoising comparison.
    
    The original data is z-scored for visualization, allowing meaningful
    comparison with the denoised data on the same scale.
    """
    original_data = original_img.get_fdata()
    denoised_data = denoised_img.get_fdata()
    
    if mask_img is not None:
        mask_data = mask_img.get_fdata().astype(bool)
    else:
        temporal_std = np.std(original_data, axis=-1)
        mask_data = temporal_std > 0
    
    original_masked = original_data[mask_data].flatten()
    denoised_masked = denoised_data[mask_data].flatten()
    
    original_masked = original_masked[~np.isnan(original_masked)]
    denoised_masked = denoised_masked[~np.isnan(denoised_masked)]
    
    if subsample > 1 and len(original_masked) > 100000:
        original_masked = original_masked[::subsample]
        denoised_masked = denoised_masked[::subsample]
    
    original_mean = np.mean(original_masked)
    original_std = np.std(original_masked)
    original_zscored = (original_masked - original_mean) / (original_std + 1e-10)
    
    original_stats = {
        'mean': float(np.mean(original_zscored)),
        'std': float(np.std(original_zscored)),
        'min': float(np.min(original_zscored)),
        'max': float(np.max(original_zscored)),
        'n_values': len(original_zscored),
        'raw_mean': float(original_mean),
        'raw_std': float(original_std)
    }
    
    denoised_stats = {
        'mean': float(np.mean(denoised_masked)),
        'std': float(np.std(denoised_masked)),
        'min': float(np.min(denoised_masked)),
        'max': float(np.max(denoised_masked)),
        'n_values': len(denoised_masked)
    }
    
    return {
        'original_data': original_zscored,
        'denoised_data': denoised_masked,
        'original_stats': original_stats,
        'denoised_stats': denoised_stats
    }


def _validate_existing_denoised_file(
    output_path: Path,
    confound_names: List[str],
    high_pass: float,
    low_pass: float,
    logger: logging.Logger,
) -> None:
    """Validate that existing denoised file matches current parameters."""
    sidecar_path = output_path.with_suffix('').with_suffix('.json')
    
    if not sidecar_path.exists():
        raise PreprocessingError(
            f"Cannot validate existing denoised file '{output_path.name}': "
            f"JSON sidecar not found. Delete existing file or set --overwrite to reprocess."
        )
    
    try:
        with sidecar_path.open('r') as f:
            existing_params = json.load(f)
    except json.JSONDecodeError as e:
        raise PreprocessingError(
            f"Cannot validate existing denoised file '{output_path.name}': "
            f"JSON sidecar is corrupted ({e}). Delete existing file or set --overwrite to reprocess."
        )
    
    expected_params = {
        'Confounds': sorted(confound_names),
        'HighPass_Hz': high_pass,
        'LowPass_Hz': low_pass,
    }
    
    existing_confounds = sorted(existing_params.get('Confounds', []))
    existing_high_pass = existing_params.get('HighPass_Hz')
    existing_low_pass = existing_params.get('LowPass_Hz')
    
    mismatches = []
    if existing_confounds != expected_params['Confounds']:
        mismatches.append(f"  - Confounds: existing={existing_confounds}, requested={expected_params['Confounds']}")
    if existing_high_pass != expected_params['HighPass_Hz']:
        mismatches.append(f"  - HighPass_Hz: existing={existing_high_pass}, requested={expected_params['HighPass_Hz']}")
    if existing_low_pass != expected_params['LowPass_Hz']:
        mismatches.append(f"  - LowPass_Hz: existing={existing_low_pass}, requested={expected_params['LowPass_Hz']}")
    
    if mismatches:
        mismatch_details = "\n".join(mismatches)
        raise PreprocessingError(
            f"Existing denoised file '{output_path.name}' was created with different parameters!\n\n"
            f"Parameter mismatches:\n{mismatch_details}\n\n"
            f"To resolve: delete existing file or set --overwrite to force reprocessing."
        )
    
    logger.debug(f"  Validated existing denoised file parameters match current configuration")
