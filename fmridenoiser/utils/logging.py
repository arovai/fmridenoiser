"""Logging configuration and utilities for fMRI Denoiser."""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional
from colorama import Fore, Style, init


# Initialize colorama for cross-platform color support
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record):
        """Format log record with color."""
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        result = super().format(record)
        record.levelname = original_levelname
        return result


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with color support.
    
    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger('fmridenoiser')
    logger.setLevel(level)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = ColoredFormatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        plain_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


@contextmanager
def timer(logger: logging.Logger, message: str):
    """Context manager for timing operations."""
    start = time.time()
    logger.info(f"Starting: {message}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {message} ({elapsed:.2f}s)")


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a section header."""
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
