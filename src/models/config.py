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
    use_backend_api: Optional[bool] = None
    extra: Optional[Dict[str, Any]] = None
    
    def should_use_backend_api(self) -> bool:
        """Check if backend API should be used for this model.
        
        Checks in order:
        1. If running inside backend server (BACKEND_SERVER_MODE env var), always return False
        2. Model-specific config (use_backend_api field)
        3. Environment variable (USE_BACKEND_API)
        4. Default to False (direct model access)
        """
        if os.getenv("BACKEND_SERVER_MODE", "false").lower() in ("true", "1", "yes", "on"):
            return False
        
        if self.use_backend_api is not None:
            return self.use_backend_api
        
        use_backend = os.getenv("USE_BACKEND_API", "false").lower().strip()
        return use_backend in ("true", "1", "yes", "on")
    
    def get_backend_api_url(self) -> str:
        """Get backend API server URL when using backend API mode.
        
        Checks environment variables in order:
        1. API_SERVER_URL
        2. BACKEND_URL
        3. Default to http://localhost:8000
        
        Returns:
            Backend API server URL
        """
        load_dotenv()
        api_url = os.getenv("API_SERVER_URL") or os.getenv("BACKEND_URL", "http://localhost:8000")
        return api_url.rstrip("/")
    
    def get_url(self) -> str:
        """Get URL from model config or environment variable.
        
        For vLLM servers, automatically ensures /v1 is included in the URL.
        """
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
                # Fallback: try generic MODEL_URL
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

