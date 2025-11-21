"""Pydantic models for model configuration."""

import os
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv


class ModelEval(BaseModel):
    """Model evaluation configuration."""
    name: str
    url: Optional[str] = None  
    batch_size: int
    prompt: str
    max_workers: int
    num_samples: int
    active: bool
    save_results: bool
    results_dir: str
    extra: Optional[Dict[str, Any]] = None
    
    def get_url(self) -> str:
        """Get URL from model config or environment variable."""
        if self.url:
            return self.url
        env_var_name = self.name.replace("/", "_").replace("-", "_").upper() + "_URL"
        load_dotenv()
        url_from_env = os.getenv(env_var_name)
        if url_from_env:
            return url_from_env
        # Fallback: try generic MODEL_URL
        url_from_env = os.getenv("MODEL_URL")
        if url_from_env:
            return url_from_env
        raise ValueError(
            f"URL not found for model '{self.name}'. "
            f"Provide it in config or set environment variable '{env_var_name}' or 'MODEL_URL'"
        )


class Config(BaseModel):
    """Root configuration containing models."""
    ocr_models: List[ModelEval]

