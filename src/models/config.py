"""Pydantic models for model configuration."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ModelEval(BaseModel):
    """Model evaluation configuration."""
    name: str
    url: str
    batch_size: int
    prompt: str
    max_workers: int
    num_samples: int
    active: bool
    save_results: bool
    results_dir: str
    extra: Optional[Dict[str, Any]] = None


class Config(BaseModel):
    """Root configuration containing OCR models."""
    ocr_models: List[ModelEval]

