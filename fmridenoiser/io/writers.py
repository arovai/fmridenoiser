"""File writers for outputs."""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


def save_nifti_with_sidecar(
    img: nib.Nifti1Image,
    output_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """Save NIfTI image with JSON sidecar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, output_path)
    
    metadata_with_timestamp = metadata.copy()
    metadata_with_timestamp['CreationTime'] = datetime.now().isoformat()
    
    sidecar_path = output_path.with_suffix('').with_suffix('.json')
    with sidecar_path.open('w') as f:
        json.dump(metadata_with_timestamp, f, indent=2)


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """Save dictionary as JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_serializable = _make_serializable(data)
    with output_path.open('w') as f:
        json.dump(data_serializable, f, indent=2)


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    else:
        return obj
