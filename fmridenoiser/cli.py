"""Command-line interface for fmridenoiser."""

import argparse
import sys
import textwrap
from pathlib import Path
from fmridenoiser.core.version import __version__


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ColoredHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter with colored output."""

    def __init__(self, prog, indent_increment=2, max_help_position=40, width=100):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = f'{Colors.BOLD}Usage:{Colors.END} '
        return super()._format_usage(usage, actions, groups, prefix)

    def start_section(self, heading):
        if heading:
            heading = f'{Colors.BOLD}{Colors.CYAN}{heading}{Colors.END}'
        super().start_section(heading)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""

    description = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           fMRI DENOISER v{__version__}                              ║
    ║            fMRI Denoising BIDS App for fMRIPrep Outputs                     ║
    ╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

    {Colors.BOLD}Description:{Colors.END}
      fmridenoiser applies denoising (confound regression + temporal filtering)
      to fMRI data preprocessed with fMRIPrep. It can also perform FD-based
      temporal censoring (motion scrubbing).

      Output denoised files are BIDS-compliant and can be used as input to
      downstream tools like connectomix for connectivity analysis.

    {Colors.BOLD}Workflow:{Colors.END}
      1. Discover preprocessed fMRI files from fMRIPrep
      2. Check geometric consistency across subjects
      3. Resample to common space if needed
      4. Apply denoising (confound regression + bandpass filtering)
      5. Optionally apply FD-based temporal censoring
      6. Save BIDS-compliant denoised outputs with JSON sidecars
    """)

    epilog = textwrap.dedent(f"""
    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}EXAMPLES{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}

    {Colors.BOLD}Basic Usage:{Colors.END}

      {Colors.YELLOW}# Denoise all subjects with default settings{Colors.END}
      fmridenoiser /data/bids /data/derivatives/denoised participant

      {Colors.YELLOW}# Denoise a specific subject{Colors.END}
      fmridenoiser /data/bids /data/derivatives/denoised participant \\
          --participant-label 01

    {Colors.BOLD}Using Denoising Strategies:{Colors.END}

      {Colors.YELLOW}# Use minimal denoising (motion parameters only){Colors.END}
      fmridenoiser /data/bids /data/output participant --strategy minimal

      {Colors.YELLOW}# Use CSF+WM with 6 motion parameters{Colors.END}
      fmridenoiser /data/bids /data/output participant --strategy csfwm_6p

      {Colors.YELLOW}# Use scrubbing strategy with FD-based censoring{Colors.END}
      fmridenoiser /data/bids /data/output participant --strategy scrubbing5

    {Colors.BOLD}Specifying fMRIPrep Location:{Colors.END}

      {Colors.YELLOW}# When fMRIPrep output is not in default location{Colors.END}
      fmridenoiser /data/bids /data/output participant \\
          --derivatives fmriprep=/data/derivatives/fmriprep

    {Colors.BOLD}Motion Censoring:{Colors.END}

      {Colors.YELLOW}# Apply FD-based censoring at 0.5 cm threshold{Colors.END}
      fmridenoiser /data/bids /data/output participant \\
          --strategy csfwm_6p --fd-threshold 0.5

      {Colors.YELLOW}# Scrub with minimum segment length of 5{Colors.END}
      fmridenoiser /data/bids /data/output participant \\
          --strategy csfwm_6p --fd-threshold 0.5 --scrub 5

      {Colors.YELLOW}# Drop dummy scans{Colors.END}
      fmridenoiser /data/bids /data/output participant --drop-initial 4

    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}DENOISING STRATEGIES{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}

      {Colors.CYAN}minimal{Colors.END}        6 motion parameters only
      {Colors.CYAN}csfwm_6p{Colors.END}       CSF + WM + 6 motion parameters
      {Colors.CYAN}csfwm_12p{Colors.END}      CSF + WM + 12 motion params (6 + derivatives)
      {Colors.CYAN}gs_csfwm_6p{Colors.END}    Global signal + CSF + WM + 6 motion params
      {Colors.CYAN}gs_csfwm_12p{Colors.END}   Global signal + CSF + WM + 12 motion params
      {Colors.CYAN}csfwm_24p{Colors.END}      CSF + WM + 24 motion params (6 + deriv + squares)
      {Colors.CYAN}compcor_6p{Colors.END}     6 aCompCor components + 6 motion params
      {Colors.CYAN}simpleGSR{Colors.END}      Global + CSF + WM + 24 motion (preserves time series)
      {Colors.CYAN}scrubbing5{Colors.END}     CSF/WM derivatives + 24 motion + FD=0.5cm + scrub=5

    {Colors.BOLD}Note:{Colors.END} 'scrubbing5' includes FD censoring and cannot be combined with
    --fd-threshold or --scrub.

    {Colors.BOLD}{Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}
    {Colors.BOLD}MORE INFORMATION{Colors.END}
    {Colors.GREEN}═══════════════════════════════════════════════════════════════════════════════{Colors.END}

      Documentation:  https://github.com/ln2t/fmridenoiser
      Report Issues:  https://github.com/ln2t/fmridenoiser/issues
      Version:        {__version__}
    """)

    parser = argparse.ArgumentParser(
        prog="fmridenoiser",
        description=description,
        epilog=epilog,
        formatter_class=ColoredHelpFormatter,
        add_help=False,
    )

    # =========================================================================
    # REQUIRED ARGUMENTS
    # =========================================================================
    required = parser.add_argument_group(
        f'{Colors.BOLD}Required Arguments{Colors.END}'
    )

    required.add_argument(
        "bids_dir",
        type=Path,
        metavar="BIDS_DIR",
        help="Path to the BIDS dataset root directory.",
    )

    required.add_argument(
        "output_dir",
        type=Path,
        metavar="OUTPUT_DIR",
        help="Path to output directory for fmridenoiser derivatives.",
    )

    required.add_argument(
        "analysis_level",
        choices=["participant"],
        metavar="{participant}",
        help="Analysis level. Currently only 'participant' is supported.",
    )

    # =========================================================================
    # GENERAL OPTIONS
    # =========================================================================
    general = parser.add_argument_group(
        f'{Colors.BOLD}General Options{Colors.END}'
    )

    general.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )

    general.add_argument(
        "--version",
        action="version",
        version=f"fmridenoiser {__version__}",
        help="Show program version and exit.",
    )

    general.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level logging).",
    )

    general.add_argument(
        "-c", "--config",
        type=Path,
        metavar="FILE",
        help="Path to configuration file (.json, .yaml, or .yml). "
             "CLI arguments override config file settings.",
    )

    # =========================================================================
    # DERIVATIVES OPTIONS
    # =========================================================================
    derivatives = parser.add_argument_group(
        f'{Colors.BOLD}Derivatives Options{Colors.END}'
    )

    derivatives.add_argument(
        "-d", "--derivatives",
        action="append",
        metavar="NAME=PATH",
        dest="derivatives",
        help="Specify location of BIDS derivatives. Format: name=path "
             "(e.g., fmriprep=/data/derivatives/fmriprep). Can be specified "
             "multiple times.",
    )

    # =========================================================================
    # BIDS FILTERS
    # =========================================================================
    filters = parser.add_argument_group(
        f'{Colors.BOLD}BIDS Entity Filters{Colors.END}',
        "Filter which data to process based on BIDS entities."
    )

    filters.add_argument(
        "-p", "--participant-label",
        metavar="LABEL",
        dest="participant_label",
        nargs='+',
        help="Process one or more participants (without 'sub-' prefix).",
    )

    filters.add_argument(
        "-t", "--task",
        metavar="TASK",
        help="Process only this task (without 'task-' prefix).",
    )

    filters.add_argument(
        "-s", "--session",
        metavar="SESSION",
        help="Process only this session (without 'ses-' prefix).",
    )

    filters.add_argument(
        "-r", "--run",
        metavar="RUN",
        type=int,
        help="Process only this run number.",
    )

    filters.add_argument(
        "--space",
        metavar="SPACE",
        help="Process only data in this template space "
             "(e.g., 'MNI152NLin2009cAsym').",
    )

    filters.add_argument(
        "--label",
        metavar="STRING",
        help="Custom label added to all output filenames (BIDS entity).",
    )

    # =========================================================================
    # DENOISING OPTIONS
    # =========================================================================
    preproc = parser.add_argument_group(
        f'{Colors.BOLD}Denoising Options{Colors.END}'
    )

    preproc.add_argument(
        "--strategy",
        metavar="STRATEGY",
        choices=["minimal", "csfwm_6p", "csfwm_12p", "gs_csfwm_6p",
                 "gs_csfwm_12p", "csfwm_24p", "compcor_6p", "simpleGSR",
                 "scrubbing5"],
        help="Use a predefined denoising strategy. See DENOISING STRATEGIES "
             "section for details.",
    )

    preproc.add_argument(
        "--high-pass",
        metavar="HZ",
        type=float,
        dest="high_pass",
        help="High-pass filter cutoff in Hz (default: 0.008).",
    )

    preproc.add_argument(
        "--low-pass",
        metavar="HZ",
        type=float,
        dest="low_pass",
        help="Low-pass filter cutoff in Hz (default: 0.1).",
    )

    preproc.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing denoised files.",
    )

    # =========================================================================
    # TEMPORAL CENSORING OPTIONS (FD-based only)
    # =========================================================================
    censoring = parser.add_argument_group(
        f'{Colors.BOLD}Temporal Censoring Options{Colors.END}',
        "FD-based motion censoring. Disabled by default."
    )

    censoring.add_argument(
        "--fd-threshold",
        metavar="CM",
        type=float,
        dest="fd_threshold",
        help="Enable motion censoring: remove volumes with framewise displacement "
             "above this threshold (in cm). fMRIPrep reports FD in cm. "
             "Typical values: 0.2-0.5 cm.",
    )

    censoring.add_argument(
        "--fd-extend",
        metavar="N",
        type=int,
        dest="fd_extend",
        default=0,
        help="Also censor N volumes before AND after high-motion volumes "
             "(default: 0).",
    )

    censoring.add_argument(
        "--scrub",
        metavar="N",
        type=int,
        dest="scrub",
        default=0,
        help="Minimum contiguous segment length to keep after motion censoring. "
             "Requires --fd-threshold.",
    )

    censoring.add_argument(
        "--drop-initial",
        metavar="N",
        type=int,
        dest="drop_initial",
        default=0,
        help="Number of initial volumes to drop (dummy scans). Default: 0.",
    )

    return parser


def parse_derivatives_arg(derivatives_list: list) -> dict:
    """Parse derivatives arguments into dictionary."""
    if not derivatives_list:
        return {}

    derivatives_dict = {}
    for derivative_arg in derivatives_list:
        if "=" not in derivative_arg:
            raise ValueError(
                f"Invalid derivatives argument: {derivative_arg}. "
                f"Expected format: name=path (e.g., fmriprep=/path/to/fmriprep)"
            )
        name, path = derivative_arg.split("=", 1)
        derivatives_dict[name] = Path(path)

    return derivatives_dict
