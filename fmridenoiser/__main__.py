"""Main entry point for fmridenoiser."""

import sys
import logging
from pathlib import Path

from fmridenoiser.cli import create_parser, parse_derivatives_arg
from fmridenoiser.config.defaults import DenoisingConfig, CensoringConfig, MotionCensoringConfig
from fmridenoiser.config.loader import load_config_file, merge_configs
from fmridenoiser.config.strategies import get_denoising_strategy, DENOISING_STRATEGIES
from fmridenoiser.utils.logging import setup_logging
from fmridenoiser.utils.exceptions import FmriDenoiserError


def main():
    """Main entry point for fmridenoiser CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)

    try:
        # Build config from args and config file
        config = _build_config(args, logger)

        # Parse derivatives
        derivatives = parse_derivatives_arg(args.derivatives) if args.derivatives else None

        if args.analysis_level == "participant":
            from fmridenoiser.core.pipeline import run_denoising_pipeline

            run_denoising_pipeline(
                bids_dir=args.bids_dir,
                output_dir=args.output_dir,
                config=config,
                derivatives=derivatives,
                logger=logger,
            )
        else:
            logger.error(f"Unknown analysis level: {args.analysis_level}")
            sys.exit(1)

    except FmriDenoiserError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def _build_config(args, logger: logging.Logger) -> DenoisingConfig:
    """Build DenoisingConfig from CLI args and optional config file.

    CLI arguments always override config file settings.
    """
    # Start with default config
    config = DenoisingConfig()

    # Load config file if provided
    if args.config:
        file_config = load_config_file(args.config)
        config = merge_configs(config, file_config)
        logger.info(f"Loaded configuration from {args.config}")

    # Apply denoising strategy if specified
    if args.strategy:
        strategy = get_denoising_strategy(args.strategy)

        # Check for conflicts with scrubbing5
        if strategy.is_rigid:
            if args.fd_threshold is not None or args.scrub:
                logger.error(
                    f"Strategy '{args.strategy}' cannot be combined with "
                    f"--fd-threshold or --scrub (it has its own censoring settings)."
                )
                sys.exit(1)

        config.denoising_strategy = args.strategy
        config.confounds = strategy.confounds

        # Apply strategy censoring if present (e.g. scrubbing5)
        if strategy.fd_threshold is not None:
            config.censoring = CensoringConfig(
                enabled=True,
                motion_censoring=MotionCensoringConfig(
                    enabled=True,
                    fd_threshold=strategy.fd_threshold,
                    min_segment_length=strategy.min_segment_length,
                ),
            )

    # CLI overrides for BIDS filters
    if args.participant_label:
        config.subject = args.participant_label

    if args.task:
        config.tasks = [args.task]

    if args.session:
        config.sessions = [args.session]

    if args.run:
        config.runs = [str(args.run)]

    if args.space:
        config.spaces = [args.space]

    if args.label:
        config.label = args.label

    # CLI overrides for denoising
    if args.high_pass is not None:
        config.high_pass = args.high_pass

    if args.low_pass is not None:
        config.low_pass = args.low_pass

    if hasattr(args, 'overwrite') and args.overwrite:
        config.overwrite = True

    # CLI overrides for censoring
    _configure_censoring(args, config, logger)

    return config


def _configure_censoring(args, config: DenoisingConfig, logger: logging.Logger) -> None:
    """Configure FD-based temporal censoring from CLI arguments."""
    # Check if any censoring-related args were provided
    has_fd = args.fd_threshold is not None
    has_scrub = args.scrub and args.scrub > 0
    has_drop = args.drop_initial and args.drop_initial > 0

    if not has_fd and not has_drop and not config.censoring.enabled:
        return

    # Enable censoring
    config.censoring.enabled = True

    # Drop initial volumes
    if has_drop:
        config.censoring.drop_initial_volumes = args.drop_initial

    # FD-based motion censoring
    if has_fd:
        config.censoring.motion_censoring.enabled = True
        config.censoring.motion_censoring.fd_threshold = args.fd_threshold

        if args.fd_extend:
            config.censoring.motion_censoring.extend_before = args.fd_extend
            config.censoring.motion_censoring.extend_after = args.fd_extend

        if has_scrub:
            config.censoring.motion_censoring.min_segment_length = args.scrub

    elif has_scrub:
        logger.warning("--scrub requires --fd-threshold to be set. Ignoring --scrub.")


if __name__ == "__main__":
    main()
