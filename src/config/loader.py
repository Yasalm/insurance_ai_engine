"""Configuration loader for YAML config files."""

import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.config import Config, ModelEval


def load_config(config_path: str = "models.yaml") -> "Config":
    """
    Load and parse configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config instance containing only active models
    """
    from ..models.config import Config, ModelEval
    
    active_models = []
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    for model in config["ocr_models"]:
        if model["active"]:
            active_models.append(ModelEval(**model))
    
    return Config(ocr_models=active_models)

