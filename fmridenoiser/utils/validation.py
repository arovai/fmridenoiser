"""Input validation functions."""

from pathlib import Path
from typing import Any, List, Optional


def validate_alpha(value: float, name: str = "alpha") -> None:
    """Validate alpha value is in [0, 1]."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate_positive(value: float, name: str = "value") -> None:
    """Validate value is positive."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str = "value") -> None:
    """Validate value is non-negative."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_file_exists(path: Path, name: str = "file") -> None:
    """Validate file exists."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")


def validate_dir_exists(path: Path, name: str = "directory") -> None:
    """Validate directory exists."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_dir():
        raise ValueError(f"{name} is not a directory: {path}")


def validate_choice(value: Any, choices: List[Any], name: str = "parameter") -> None:
    """Validate value is in allowed choices."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got '{value}'")
