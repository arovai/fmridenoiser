"""File readers for various formats."""

import fnmatch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json


def expand_confound_wildcards(
    confound_patterns: List[str],
    available_columns: List[str],
) -> List[str]:
    """Expand wildcard patterns to match actual confound column names."""
    expanded = []
    seen = set()
    
    for pattern in confound_patterns:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            matches = sorted(fnmatch.filter(available_columns, pattern))
            for match in matches:
                if match not in seen:
                    expanded.append(match)
                    seen.add(match)
        else:
            if pattern not in seen:
                expanded.append(pattern)
                seen.add(pattern)
    
    return expanded


def load_confounds(
    confounds_path: Path,
    confound_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Load and extract confounds from TSV file.
    
    Supports wildcard patterns in confound names.
    
    Args:
        confounds_path: Path to fMRIPrep confounds TSV file
        confound_names: List of confound column names or wildcard patterns
    
    Returns:
        Tuple of (confounds_array, expanded_names)
    """
    if not confounds_path.exists():
        raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
    
    df = pd.read_csv(confounds_path, sep='\t')
    expanded_names = expand_confound_wildcards(confound_names, df.columns.tolist())
    
    # Check patterns had matches
    for pattern in confound_names:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            matches = fnmatch.filter(df.columns.tolist(), pattern)
            if not matches:
                available = sorted(df.columns.tolist())
                raise ValueError(
                    f"No confounds matching pattern '{pattern}' in {confounds_path.name}.\n"
                    f"  Available columns: {available[:15]}{'...' if len(available) > 15 else ''}"
                )
        else:
            if pattern not in df.columns:
                available = sorted(df.columns.tolist())
                similar = [c for c in available if pattern.lower() in c.lower()]
                suggestion = f"\n  Similar columns: {similar[:5]}" if similar else ""
                raise ValueError(
                    f"Confound '{pattern}' not found in {confounds_path.name}.{suggestion}\n"
                    f"  Available columns: {available[:15]}{'...' if len(available) > 15 else ''}"
                )
    
    if not expanded_names:
        raise ValueError(f"No confounds selected after expanding patterns: {confound_names}")
    
    confounds = df[expanded_names].values
    
    # Handle NaN values
    if np.any(np.isnan(confounds)):
        col_means = np.nanmean(confounds, axis=0)
        nan_mask = np.isnan(confounds)
        confounds[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    return confounds, expanded_names


def load_json_sidecar(json_path: Path) -> Dict[str, Any]:
    """Load JSON sidecar file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open() as f:
        return json.load(f)


def get_repetition_time(json_path: Path) -> float:
    """Get repetition time (TR) from JSON sidecar."""
    data = load_json_sidecar(json_path)
    if 'RepetitionTime' not in data:
        raise ValueError(f"RepetitionTime not found in {json_path.name}")
    return float(data['RepetitionTime'])
