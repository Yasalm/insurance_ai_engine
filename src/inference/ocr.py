"""OCR inference functions supporting both backend API server and direct model access."""

import os
import time
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import httpx
from ..models.config import ModelEval

logger = logging.getLogger(__name__)






def infer(
    img_base64: str,
    model_config: ModelEval,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Run OCR inference on a single image with retry logic.
    
    Supports two modes:
    - Backend API mode: Calls FastAPI backend server endpoint (set USE_BACKEND_API=true)
    - Direct mode: Calls model directly via OpenAI-compatible API (default)

    Args:
        img_base64: Base64 encoded image string
        model_config: ModelEval configuration instance
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        OCR text result

    Raises:
        Exception: If inference fails after all retries
    """
    use_backend = model_config.should_use_backend_api()
    logger.info(
        f"OCR inference mode for {model_config.name}: "
        f"{'Backend API' if use_backend else 'Direct model access'}"
    )
    
    if use_backend:
        return _infer_via_backend(img_base64, model_config, max_retries, retry_delay)
    else:
        return _infer_direct(img_base64, model_config, max_retries, retry_delay)


def _infer_via_backend(
    img_base64: str,
    model_config: ModelEval,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Run OCR inference via backend API server."""
    api_url = model_config.get_backend_api_url()
    endpoint = f"{api_url}/ocr/infer-base64"
    logger.info(f"Calling backend API endpoint: {endpoint} for model {model_config.name}")
    
    for attempt in range(max_retries):
        try:
            response = httpx.post(
                endpoint,
                json={
                    "image_base64": img_base64,
                    "model_name": model_config.name,
                    "max_retries": 1, 
                },
                timeout=httpx.Timeout(1800.0, connect=60.0), 
            )
            response.raise_for_status()
            result = response.json()
            return result.get("text", "")
        except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout) as e:
            error_type = type(e).__name__
            if attempt < max_retries - 1:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"Backend API connection failed (attempt {attempt + 1}/{max_retries}): {error_type}. "
                    f"Ensure backend server is running at {api_url}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Backend API connection failed after {max_retries} attempts: {error_type}: {e}. "
                    f"Backend server may not be running at {api_url}. "
                    f"Start it with: make start-api-server or set API_SERVER_URL environment variable."
                )
                raise
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Backend API returned error status {e.response.status_code}: {e.response.text}"
            )
            raise
        except Exception as e:
            error_type = type(e).__name__
            if attempt < max_retries - 1:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"Inference failed (attempt {attempt + 1}/{max_retries}): {error_type}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Inference failed after {max_retries} attempts: {error_type}: {e}"
                )
                raise

def _infer_direct(
    img_base64: str,
    model_config: ModelEval,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Run OCR inference directly via OpenAI-compatible API."""
    model_url = model_config.get_url()
    logger.debug(f"Calling model directly at: {model_url}")
    client = OpenAI(
        base_url=model_url,
        api_key=os.getenv("OPENAI_API_KEY", "DUMMY_API_KEY"),
        timeout=httpx.Timeout(1800.0, connect=60.0),  # 30 mins total, 60s connect
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_config.name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                },
                            },
                            {"type": "text", "text": model_config.prompt},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=4000,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_type = type(e).__name__
            if attempt < max_retries - 1:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"Inference failed (attempt {attempt + 1}/{max_retries}): {error_type}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Inference failed after {max_retries} attempts: {error_type}: {e}"
                )
                raise


def infer_batch(
    img_base64_list: List[str], 
    model_config: ModelEval, 
    max_workers: int = 5,
) -> List[str]:
    """
    Run OCR inference on multiple images in parallel.

    Args:
        img_base64_list: List of base64 encoded image strings
        model_config: ModelEval configuration instance
        max_workers: Maximum number of parallel workers

    Returns:
        List of OCR text results (empty strings for failed images)
    """
    results = [""] * len(img_base64_list)

    def process_single(idx, img_base64):
        if img_base64 is None:
            return idx, ""
        try:
            return idx, infer(img_base64, model_config)
        except Exception:
            logger.warning(f"Failed to process image {idx}")
            return idx, ""

    valid_items = [
        (idx, img) for idx, img in enumerate(img_base64_list) if img is not None
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, idx, img): idx for idx, img in valid_items
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results
