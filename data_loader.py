from datasets import load_dataset
import os


def download_dataset(
    dataset_name: str = "amaye15/invoices-google-ocr",
    cache_dir: str = "./data",
    split: str = None,
):
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split if split else None, cache_dir=cache_dir)
    return dataset