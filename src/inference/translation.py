"""Translation inference functions for mBART and LLM models."""

import os
import time
import logging
import threading
from functools import lru_cache
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import httpx
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

from ..models.config import TranslationModel

logger = logging.getLogger(__name__)
_model_lock = threading.Lock() # because we're running on concurrent requests and this is to ensure that only one request at a time loads the model


@lru_cache(maxsize=None)
def _load_mbart_model(
    model_path: str,
) -> Tuple[MBartForConditionalGeneration, MBart50TokenizerFast]:
    logger.info(f"Loading mBART model: {model_path}")
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MBartForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,
    )
    
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded mBART model: {model_path} on device: {device}")
    return model, tokenizer


def preload_mbart_model(model_config: TranslationModel) -> bool:
    """
    Preload an mBART model into memory.
    
    Args:
        model_config: TranslationModel configuration instance
        
    Returns:
        True if model was loaded successfully, False otherwise
    """
    if model_config.type != "mbart":
        logger.warning(f"Skipping preload for non-mBART model: {model_config.name} (type: {model_config.type})")
        return False
    
    model_path = model_config.model_path or model_config.name
    if not model_path:
        logger.warning(f"Model '{model_config.name}' has no model_path specified, skipping preload")
        return False
    
    try:
        logger.info(f"Preloading mBART model: {model_config.name} (path: {model_path})")
        _load_mbart_model(model_path)
        logger.info(f"Successfully preloaded mBART model: {model_config.name}")
        return True
    except Exception as e:
        logger.error(
            f"Failed to preload mBART model '{model_config.name}': {type(e).__name__}: {e}",
            exc_info=True
        )
        return False


def infer_mbart(
    text: str,
    model_config: TranslationModel,
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    max_length: int = 50,
    num_beams: int = 4,
) -> str:
    """
    Run translation inference using mBART model directly with Transformers.

    Args:
        text: Input text to translate
        model_config: TranslationModel configuration instance
        src_lang: Source language code (e.g., "en_XX"). Uses config default if None
        target_lang: Target language code (e.g., "fr_XX"). Uses config default if None
        max_length: Maximum length of generated translation

    Returns:
        Translated text

    Raises:
        ValueError: If model_path or language codes are missing
        Exception: If inference fails
    """
    if model_config.type != "mbart":
        raise ValueError(
            f"infer_mbart() requires mbart model type, got {model_config.type}"
        )

    model_path = model_config.model_path or model_config.name
    if not model_path:
        raise ValueError(f"model_path required for mBART model '{model_config.name}'")

    src_lang = src_lang or model_config.src_lang
    target_lang = target_lang or model_config.target_lang

    if not src_lang or not target_lang:
        raise ValueError(
            f"Source and target languages required for mBART model '{model_config.name}'. "
            f"Provide in config or as parameters."
        )

    try:
        with _model_lock:
            model, tokenizer = _load_mbart_model(model_path)
            device = next(model.parameters()).device
            tokenizer.src_lang = src_lang
            encoded_input = tokenizer(text, return_tensors="pt")
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
            if isinstance(forced_bos_token_id, torch.Tensor):
                forced_bos_token_id = forced_bos_token_id.item()

            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded_input,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            translation = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0]
            return translation

    except Exception as e:
        logger.error(f"mBART inference failed for model '{model_config.name}': {e}")
        raise


def infer_llm(
    text: str,
    model_config: TranslationModel,
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Run translation inference using LLM model via OpenAI-compatible API.

    Args:
        text: Input text to translate
        model_config: TranslationModel configuration instance
        src_lang: Source language name (e.g., "English"). Uses config default if None
        target_lang: Target language name (e.g., "French"). Uses config default if None
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)

    Returns:
        Translated text

    Raises:
        ValueError: If model type is not "llm"
        Exception: If inference fails after all retries
    """
    if model_config.type != "llm":
        raise ValueError(
            f"infer_llm() requires llm model type, got {model_config.type}"
        )

    src_lang = src_lang or model_config.src_lang or "English"
    target_lang = target_lang or model_config.target_lang or "French"
    prompt = f"Translate the following text from {src_lang} to {target_lang}:\n\n{text}"

    client = OpenAI(
        base_url=model_config.get_url(),
        api_key=os.getenv("OPENAI_API_KEY", "DUMMY_API_KEY"),
        timeout=httpx.Timeout(1800.0, connect=60.0),
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_config.name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_type = type(e).__name__
            if attempt < max_retries - 1:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"Translation inference failed (attempt {attempt + 1}/{max_retries}): {error_type}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Translation inference failed after {max_retries} attempts: {error_type}: {e}"
                )
                raise


def infer(
    text: str,
    model_config: TranslationModel,
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Run translation inference on a single text (dispatches to mBART or LLM based on model type).

    Args:
        text: Input text to translate
        model_config: TranslationModel configuration instance
        src_lang: Source language (code for mBART, name for LLM). Uses config default if None
        target_lang: Target language (code for mBART, name for LLM). Uses config default if None
        max_retries: Maximum number of retry attempts (for LLM models)
        retry_delay: Initial delay between retries (for LLM models)

    Returns:
        Translated text

    Raises:
        ValueError: If model type is unknown
        Exception: If inference fails
    """
    if model_config.type == "mbart":
        return infer_mbart(text, model_config, src_lang, target_lang)
    elif model_config.type == "llm":
        return infer_llm(
            text, model_config, src_lang, target_lang, max_retries, retry_delay
        )
    else:
        raise ValueError(
            f"Unknown translation model type: {model_config.type}. Must be 'mbart' or 'llm'"
        )


def infer_batch(
    text_list: List[str],
    model_config: TranslationModel,
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> List[str]:
    """
    Run translation inference on multiple texts in parallel.

    Args:
        text_list: List of input texts to translate
        model_config: TranslationModel configuration instance
        src_lang: Source language (code for mBART, name for LLM). Uses config default if None
        target_lang: Target language (code for mBART, name for LLM). Uses config default if None
        max_workers: Maximum number of parallel workers (uses config default if None)

    Returns:
        List of translated texts (empty strings for failed translations)
    """
    results = [""] * len(text_list)
    workers = max_workers if max_workers is not None else model_config.max_workers

    def process_single(idx, text):
        if not text or text.strip() == "":
            return idx, ""
        try:
            return idx, infer(text, model_config, src_lang, target_lang)
        except Exception:
            logger.warning(f"Failed to translate text {idx}")
            return idx, ""

    valid_items = [
        (idx, txt) for idx, txt in enumerate(text_list) if txt and txt.strip()
    ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single, idx, txt): idx for idx, txt in valid_items
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results
