"""Configuration loader for YAML config files."""

import os
import yaml
from typing import TYPE_CHECKING

from ..models.config import Config, ModelEval, TranslationModel


def load_config(config_path: str = None) -> "Config":
    """
    Load and parse configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses models.yaml
                    from the same directory as this module.
        
    Returns:
        Config instance containing only active models
    """
    
    
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "models.yaml")
    
    active_ocr_models = []
    active_translation_models = []
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if "ocr_models" in config:
        for model in config["ocr_models"]:
            if model["active"]:
                active_ocr_models.append(ModelEval(**model))
    
    if "translation_models" in config:
        for model in config["translation_models"]:
            if model.get("active", True):
                active_translation_models.append(TranslationModel(**model))
    
    return Config(
        ocr_models=active_ocr_models,
        translation_models=active_translation_models
    )

