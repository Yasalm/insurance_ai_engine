"""Dataset loading utilities."""

from datasets import load_dataset
import os


def download_dataset(
    dataset_name: str = "amaye15/invoices-google-ocr",
    cache_dir: str = "../../data",
    split: str = None,
):
    """
    Download and load a dataset from HuggingFace.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        cache_dir: Directory to cache the dataset
        split: Dataset split to load (train, test, validation, etc.)
        
    Returns:
        Loaded dataset
    """
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split if split else None, cache_dir=cache_dir)
    return dataset

