"""Dataset loading utilities."""

from functools import lru_cache
from datasets import load_dataset
import os


@lru_cache(maxsize=None)
def download_dataset(
    dataset_name: str = "amaye15/invoices-google-ocr",
    cache_dir: str = "data",
    split: str = None,
):
    """
    Download and load a dataset from HuggingFace.
    Downloads to cache_dir on first run, loads from cache on subsequent runs.
    Uses lru_cache to cache loaded datasets in memory.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        cache_dir: Directory to cache the dataset (default: 'data')
        split: Dataset split to load (train, test, validation, etc.)
        
    Returns:
        Loaded dataset
    """
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split if split else None, cache_dir=cache_dir)
    return dataset

