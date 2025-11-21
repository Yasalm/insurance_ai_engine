"""OCR inference functions using OpenAI-compatible API."""

import os
import time
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

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

    Args:
        img_base64: Base64 encoded image string
        model_config: ModelEval configuration instance
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        OCR text result from the model

    Raises:
        Exception: If inference fails after all retries
    """
    client = OpenAI(
        base_url=model_config.url,
        api_key=os.getenv("OPENAI_API_KEY", "DUMMY_API_KEY"),
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
                max_tokens=15000,
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
    img_base64_list: List[str], model_config: ModelEval, max_workers: int = 5
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
