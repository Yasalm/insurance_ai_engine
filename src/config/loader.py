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
        url = None
        if self.url:
            url = self.url
        else:
            env_var_name = self.name.replace("/", "_").replace("-", "_").upper() + "_URL"
            load_dotenv()
            url_from_env = os.getenv(env_var_name)
            if url_from_env:
                url = url_from_env
            else:
                url_from_env = os.getenv("MODEL_URL")
                if url_from_env:
                    url = url_from_env
        if not url:
            env_var_name = self.name.replace("/", "_").replace("-", "_").upper() + "_URL"
            raise ValueError(
                f"URL not found for model '{self.name}'. "
                f"Provide it in config or set environment variable '{env_var_name}' or 'MODEL_URL'"
            )
        url = url.rstrip('/')
        if not url.endswith('/v1'):
            url = url + '/v1'
        
        return url


class TranslationModel(BaseModel):
    """Translation model configuration."""
    name: str
    type: str
    url: Optional[str] = None
    model_path: Optional[str] = None
    src_lang: Optional[str] = None
    target_lang: Optional[str] = None
    active: bool = True
    max_workers: int = 3
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
        if self.type == "mbart":
            # its not actually needed, it runs locally.
            url_from_env = os.getenv("TRANSLATION_SERVER_URL", "http://localhost:8004")
        elif self.type == "llm":
            url_from_env = os.getenv("MODEL_URL")
        else:
            url_from_env = os.getenv("MODEL_URL")
        if url_from_env:
            return url_from_env
        raise ValueError(
            f"URL not found for {self.type} model '{self.name}'. "
            f"Provide it in config or set environment variable '{env_var_name}' or 'TRANSLATION_SERVER_URL' (for mbart) or 'MODEL_URL' (for llm)"
        )


class Config(BaseModel):
    """Root configuration containing models."""
    ocr_models: List[ModelEval]
    translation_models: List[TranslationModel] = []

