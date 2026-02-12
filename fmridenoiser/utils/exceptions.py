"""Custom exceptions for fMRI Denoiser."""


class FmriDenoiserError(Exception):
    """Base exception for fMRI Denoiser."""
    pass


class BIDSError(FmriDenoiserError):
    """Error related to BIDS dataset."""
    pass


class ConfigurationError(FmriDenoiserError):
    """Error in configuration."""
    pass


class PreprocessingError(FmriDenoiserError):
    """Error during preprocessing."""
    pass
