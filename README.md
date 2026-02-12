# fmridenoiser

**fMRI Denoising BIDS App for fMRIPrep Outputs**

fmridenoiser applies denoising (confound regression + temporal filtering) and optional FD-based temporal censoring to fMRI data preprocessed with [fMRIPrep](https://fmriprep.org). It produces BIDS-compliant denoised outputs that can be used as input to downstream tools like [connectomix](https://github.com/ln2t/connectomix) (connectomix version 4.0.0 and onwards).

## Features

- **9 predefined denoising strategies** based on neuroimaging best practices (Wang et al. 2024)
- **FD-based motion censoring** with configurable threshold, extension, and segment filtering (scrubbing)
- **Geometric consistency** checking and resampling across subjects
- **BIDS-compliant** outputs with JSON sidecars for provenance tracking
- **HTML quality reports** with denoising histograms, confound time series, and FD traces
- **Brain mask copying** from fMRIPrep outputs (both anatomical and functional masks)
- **Wildcard support** for confound selection (e.g., `a_comp_cor_*`)

## Installation

```bash
git clone https://github.com/ln2t/fmridenoiser.git
cd fmridenoiser
pip install -e .
```

## Quick Start

```bash
# Basic denoising with a predefined strategy
fmridenoiser /data/bids /data/derivatives/denoised participant --strategy csfwm_6p

# Process a specific subject
fmridenoiser /data/bids /data/output participant --participant-label 01 --strategy minimal

# With FD-based motion censoring
fmridenoiser /data/bids /data/output participant --strategy csfwm_6p --fd-threshold 0.5

# Using scrubbing5 strategy (includes FD censoring)
fmridenoiser /data/bids /data/output participant --strategy scrubbing5

# Specify fMRIPrep derivatives location
fmridenoiser /data/bids /data/output participant \
    --derivatives fmriprep=/path/to/fmriprep \
    --strategy gs_csfwm_12p
```

## Denoising Strategies

| Strategy | Confounds | Description |
|----------|-----------|-------------|
| `minimal` | 6 motion params | Motion parameters only |
| `csfwm_6p` | CSF + WM + 6 motion | Standard physiological + motion |
| `csfwm_12p` | CSF + WM + 12 motion | With motion derivatives |
| `gs_csfwm_6p` | GS + CSF + WM + 6 motion | With global signal regression |
| `gs_csfwm_12p` | GS + CSF + WM + 12 motion | GSR + motion derivatives |
| `csfwm_24p` | CSF + WM + 24 motion | Full motion model |
| `compcor_6p` | 6 aCompCor + 6 motion | Data-driven + motion |
| `simpleGSR` | GS + CSF + WM + 24 motion | Preserves time series continuity |
| `scrubbing5` | CSF/WM deriv + 24 motion + FD=0.5cm + scrub=5 | Maximum denoising quality |

## Processing Order

Temporal censoring (volume removal) is applied **after** denoising. Denoising via confound regression and temporal filtering (`nilearn.image.clean_img`) is performed on the full time series first, then volumes are removed based on motion thresholds if censoring is enabled.

## Output Structure

```
output_dir/
├── dataset_description.json
├── config/
│   └── backups/
│       └── denoising_config_YYYYMMDD_HHMMSS.json
└── sub-01/
    ├── func/
    │   ├── sub-01_task-rest_space-MNI_denoise-csfwm6p_desc-denoised_bold.nii.gz
    │   └── sub-01_task-rest_space-MNI_denoise-csfwm6p_desc-denoised_bold.json
    ├── masks/
    │   ├── sub-01_space-MNI_desc-brain_mask.nii.gz
    │   └── sub-01_task-rest_space-MNI_desc-brain_mask.nii.gz
    ├── figures/
    │   └── [QA figures]
    └── sub-01_task-rest_desc-csfwm6p_denoising-report.html
```

## Integration with connectomix

fmridenoiser is designed to work as a preprocessing step before [connectomix](https://github.com/ln2t/connectomix):

```bash
# Step 1: Denoise with fmridenoiser
fmridenoiser /data/bids /data/derivatives/fmridenoiser participant --strategy csfwm_6p

# Step 2: Compute connectivity with connectomix
connectomix /data/derivatives/fmridenoiser /data/derivatives/connectomix participant \
    --method roiToRoi --atlas schaefer2018n100
```

## References

- **fMRIPrep**: Esteban et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods*, 16, 111-116.
- **Nilearn**: Abraham et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*, 8, 14.
- **Denoising strategies**: Wang et al. (2024). Benchmarking fMRI denoising strategies for functional connectomics.
- **Motion scrubbing**: Power et al. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage*, 59, 2142-2154.

## License

MIT License
