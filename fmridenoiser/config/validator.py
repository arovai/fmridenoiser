"""Configuration parameter validation."""

from typing import Any, List
from pathlib import Path


class ConfigValidator:
    """Validate configuration parameters with error accumulation."""
    
    def __init__(self):
        self.errors: List[str] = []
    
    def validate_alpha(self, value: float, name: str) -> bool:
        if not isinstance(value, (int, float)):
            self.errors.append(f"{name} must be a number, got {type(value).__name__}")
            return False
        if not 0 <= value <= 1:
            self.errors.append(f"{name} must be between 0 and 1, got {value}")
            return False
        return True
    
    def validate_positive(self, value: float, name: str) -> bool:
        if not isinstance(value, (int, float)):
            self.errors.append(f"{name} must be a number, got {type(value).__name__}")
            return False
        if value <= 0:
            self.errors.append(f"{name} must be positive, got {value}")
            return False
        return True
    
    def validate_non_negative(self, value: float, name: str) -> bool:
        if not isinstance(value, (int, float)):
            self.errors.append(f"{name} must be a number, got {type(value).__name__}")
            return False
        if value < 0:
            self.errors.append(f"{name} must be non-negative, got {value}")
            return False
        return True
    
    def validate_choice(self, value: Any, choices: List[Any], name: str) -> bool:
        if value not in choices:
            self.errors.append(f"{name} must be one of {choices}, got '{value}'")
            return False
        return True
    
    def raise_if_errors(self) -> None:
        if self.errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in self.errors
            )
            raise ValueError(error_msg)
