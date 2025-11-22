"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Optional

from ..models.config import Config


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to models.yaml file. Defaults to src/config/models.yaml
        
    Returns:
        Config instance with loaded models
    """
    if config_path is None:
        # Default to src/config/models.yaml relative to this file
        config_dir = Path(__file__).parent
        config_path = config_dir / "models.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return Config(**config_data)

