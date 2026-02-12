"""BIDS layout creation and file querying."""

from bids import BIDSLayout
from pathlib import Path
from typing import Dict, Optional, Any
import logging

from fmridenoiser.io.paths import validate_bids_dir, validate_derivatives_dir


def create_bids_layout(
    bids_dir: Path,
    derivatives: Optional[Dict[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> BIDSLayout:
    """Create BIDS layout with derivatives.
    
    Args:
        bids_dir: Path to BIDS dataset root
        derivatives: Dictionary mapping derivative names to paths
        logger: Optional logger instance
    
    Returns:
        BIDSLayout instance
    """
    validate_bids_dir(bids_dir)
    
    if logger:
        logger.info(f"Creating BIDS layout for {bids_dir}")
    
    derivatives_list = []
    
    if derivatives:
        for name, path in derivatives.items():
            validate_derivatives_dir(path, name)
            derivatives_list.append(str(path))
            if logger:
                logger.debug(f"  Adding {name} derivatives: {path}")
    else:
        default_fmriprep = bids_dir / "derivatives" / "fmriprep"
        if default_fmriprep.exists():
            derivatives_list.append(str(default_fmriprep))
            if logger:
                logger.debug(f"  Found fmriprep at default location: {default_fmriprep}")
    
    layout = BIDSLayout(
        str(bids_dir),
        derivatives=derivatives_list if derivatives_list else False,
        validate=False
    )
    
    if logger:
        n_subjects = len(layout.get_subjects())
        logger.info(f"Found {n_subjects} subject(s) in dataset")
    
    return layout


def query_participant_files(
    layout: BIDSLayout,
    entities: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Dict[str, list]:
    """Query fMRIPrep files for participant-level processing.
    
    Args:
        layout: BIDSLayout instance
        entities: Dictionary of BIDS entities for filtering
        logger: Optional logger instance
    
    Returns:
        Dictionary with keys 'func', 'json', 'confounds' containing file lists
    """
    query_params = {
        'extension': 'nii.gz',
        'suffix': 'bold',
        'desc': 'preproc',
        'scope': 'derivatives',
        'invalid_filters': 'allow',
    }
    
    if entities.get('subject'):
        query_params['subject'] = entities['subject']
    if entities.get('session'):
        query_params['session'] = entities['session']
    if entities.get('task'):
        query_params['task'] = entities['task']
    if entities.get('run'):
        query_params['run'] = entities['run']
    if entities.get('space'):
        query_params['space'] = entities['space']
    
    func_files = layout.get(**query_params)
    
    if logger:
        logger.info(f"Found {len(func_files)} functional image(s)")
    
    if len(func_files) == 0:
        relaxed_q = {k: v for k, v in query_params.items() if k != "desc"}
        try:
            candidate_files = layout.get(**relaxed_q)
        except Exception:
            candidate_files = []

        candidate_paths = [f.path for f in candidate_files][:10]

        raise ValueError(
            f"No functional images found matching criteria:\n"
            f"  {query_params}\n"
            f"Example candidate files (relaxed search without 'desc'): {candidate_paths}\n"
            f"Please check your BIDS entities and fMRIPrep outputs. If fMRIPrep "
            f"is in a non-standard location, specify it with --derivatives "
            f"(e.g., --derivatives fmriprep=/path/to/fmriprep)."
        )
    
    json_files = []
    confounds_files = []
    
    for func_file in func_files:
        json_files.append(func_file.path.replace('.nii.gz', '.json'))
        
        confounds_query = {
            'subject': func_file.entities['subject'],
            'suffix': 'timeseries',
            'desc': 'confounds',
            'extension': 'tsv',
            'scope': 'derivatives',
            'invalid_filters': 'allow',
        }
        
        for entity in ['session', 'task', 'run']:
            if entity in func_file.entities:
                confounds_query[entity] = func_file.entities[entity]
        
        confounds = layout.get(**confounds_query)
        
        if len(confounds) == 0:
            raise ValueError(f"No confounds file found for {func_file.filename}")
        
        confounds_files.append(confounds[0].path)
    
    return {
        'func': [f.path for f in func_files],
        'json': json_files,
        'confounds': confounds_files
    }
