"""BIDS dataset path validation and initialization."""

from pathlib import Path
import json
from typing import Optional
import logging

from fmridenoiser.utils.exceptions import BIDSError


def validate_bids_dir(path: Path) -> None:
    """Validate BIDS directory structure."""
    if not path.exists():
        raise BIDSError(f"BIDS directory not found: {path}\nPlease check the path and try again.")
    if not path.is_dir():
        raise BIDSError(f"BIDS path is not a directory: {path}")
    
    dataset_desc = path / "dataset_description.json"
    if not dataset_desc.exists():
        raise BIDSError(
            f"Not a valid BIDS dataset: {path}\n"
            f"Missing dataset_description.json\n"
            f"See https://bids.neuroimaging.io for BIDS specification."
        )
    
    try:
        with dataset_desc.open() as f:
            desc = json.load(f)
        if "Name" not in desc:
            raise BIDSError("Invalid dataset_description.json: missing 'Name' field")
        if "BIDSVersion" not in desc:
            raise BIDSError("Invalid dataset_description.json: missing 'BIDSVersion' field")
    except json.JSONDecodeError as e:
        raise BIDSError(f"Invalid dataset_description.json: {e}")


def create_dataset_description(output_dir: Path, version: str) -> None:
    """Create dataset_description.json for fmridenoiser derivatives."""
    dataset_desc = {
        "Name": "fMRI Denoiser",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{
            "Name": "fmridenoiser",
            "Version": version,
            "CodeURL": "https://github.com/ln2t/fmridenoiser"
        }]
    }
    
    output_path = output_dir / "dataset_description.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w') as f:
        json.dump(dataset_desc, f, indent=2)


def validate_derivatives_dir(derivatives_dir: Path, derivative_name: str = "fmriprep") -> None:
    """Validate derivatives directory."""
    if not derivatives_dir.exists():
        raise BIDSError(
            f"{derivative_name} derivatives directory not found: {derivatives_dir}\n"
            f"Please specify the correct path using:\n"
            f"  --derivatives {derivative_name}=/path/to/{derivative_name}"
        )
    if not derivatives_dir.is_dir():
        raise BIDSError(f"{derivative_name} path is not a directory: {derivatives_dir}")
    
    dataset_desc = derivatives_dir / "dataset_description.json"
    if not dataset_desc.exists():
        raise BIDSError(
            f"Not a valid {derivative_name} derivatives directory: {derivatives_dir}\n"
            f"Missing dataset_description.json"
        )
