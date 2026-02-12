"""Configuration file loading utilities."""

from pathlib import Path
from typing import Dict, Any, Type, TypeVar
import json
import yaml


ConfigType = TypeVar('ConfigType')


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file.
    
    Args:
        path: Path to configuration file
    
    Returns:
        Dictionary containing configuration parameters
    
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If file format is not supported
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open() as f:
        if path.suffix == ".json":
            return json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {path.suffix}. "
                f"Supported formats: .json, .yaml, .yml"
            )


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: Any, path: Path) -> None:
    """Save configuration to JSON file."""
    from dataclasses import asdict, is_dataclass
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_dataclass(config) and not isinstance(config, type):
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise TypeError(f"Expected dict or dataclass, got {type(config)}")
    
    config_serializable = _make_serializable(config_dict)
    
    with path.open('w') as f:
        json.dump(config_serializable, f, indent=2)


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    else:
        return obj
