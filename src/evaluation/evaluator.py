import os
import json
import time
import logging
import evaluate
from datetime import datetime
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from ..models.config import ModelEval
from ..inference.ocr import infer
from ..utils import encode_image, extract_ground_truth_text, clean_html_markdown, clean_ocr_text

logger = logging.getLogger(__name__)


def evaluate_dataset(
    dataset,
    model_config: ModelEval,
    metric: Optional[Any] = None,
    image_column: str = "pixel_values",
    ground_truth_column: str = "text",
    ocr_column: Optional[str] = "ocr",
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    num_samples: Optional[int] = None,
    save_results: Optional[bool] = None,
    results_dir: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    clean_predictions: bool = True,
    dataset_name: Optional[str] = None,
    dataset_split: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate OCR model on a dataset with comprehensive metrics.
    
    Computes CER, WER, BLEU, ROUGE, and METEOR metrics on both raw and cleaned predictions.
    
    Args:
        dataset: HuggingFace Dataset object
        model_config: ModelEval configuration instance
        metric: Optional single metric for backward compatibility (defaults to None, computes all metrics)
        image_column: Name of column containing images
        ground_truth_column: Name of column containing ground truth text (will be created if ocr_column provided)
        ocr_column: Name of column containing OCR data structure (optional, used to extract ground truth)
        batch_size: Number of samples per batch (defaults to model_config.batch_size)
        max_workers: Number of parallel API calls (defaults to model_config.max_workers)
        num_samples: Number of samples to evaluate (defaults to model_config.num_samples)
        save_results: Whether to save results to JSON file (defaults to model_config.save_results)
        results_dir: Directory to save results (defaults to model_config.results_dir)
        extra: Additional metadata to include in saved results (defaults to model_config.extra)
        clean_predictions: Whether to clean predictions using clean_ocr_text (defaults to True)
        dataset_name: Name of the dataset (optional, for tracking)
        dataset_split: Split name (e.g., 'train', 'test') (optional, for tracking)
    
    Returns:
        Dictionary containing:
        - model_config: Model configuration details
        - dataset_info: Dataset information
        - predictions: Raw and cleaned predictions
        - ground_truth: Ground truth text
        - metrics: Raw and cleaned metric scores (CER, WER, BLEU, ROUGE, METEOR)
        - processing_config: Processing parameters used
        - samples: Detailed per-sample results with bounding boxes
    """
    # Use model_config defaults if not provided
    batch_size = batch_size if batch_size is not None else model_config.batch_size
    max_workers = max_workers if max_workers is not None else model_config.max_workers
    num_samples = num_samples if num_samples is not None else model_config.num_samples
    save_results = save_results if save_results is not None else model_config.save_results
    results_dir = results_dir if results_dir is not None else model_config.results_dir
    extra = extra if extra is not None else model_config.extra
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    if ocr_column and ocr_column in dataset.column_names:
        logger.info(f"Extracting ground truth text from '{ocr_column}' column...")
        def add_text_column(example):
            ocr_data = example.get(ocr_column, [])
            example[ground_truth_column] = extract_ground_truth_text(ocr_data)
            return example
        dataset = dataset.map(add_text_column)
    
    if ground_truth_column not in dataset.column_names:
        raise ValueError(f"Ground truth column '{ground_truth_column}' not found in dataset")
    
    logger.info(f"Processing {len(dataset)} samples with batch_size={batch_size}, max_workers={max_workers}...")
    start_time = time.time()
    
    total_samples = len(dataset)
    
    progress_tracker = {"progress": None, "task_id": None, "completed": 0}
    
    def run_ocr(batch):
        images = batch[image_column]
        results = [""] * len(images)
        
        def process_image(idx, img):
            try:
                if isinstance(img, Image.Image):
                    img_rgb = img.convert("RGB")
                else:
                    img_rgb = Image.fromarray(img).convert("RGB") if hasattr(img, 'shape') else img
                
                encoded_img = encode_image(img_rgb)
                return idx, infer(encoded_img, model_config, max_retries=3)
            except Exception as e:
                logger.warning(f"Error processing image {idx}: {type(e).__name__}: {e}")
                return idx, ""
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_image, idx, img): idx for idx, img in enumerate(images)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        batch["predictions"] = results
        
        if progress_tracker["progress"] and progress_tracker["task_id"] is not None:
            progress_tracker["completed"] += len(images)
            progress_tracker["progress"].update(
                progress_tracker["task_id"], 
                completed=progress_tracker["completed"]
            )
        
        return batch
    
    batch_count = [0]
    
    def run_ocr_with_delay(batch):
        if batch_count[0] > 0:
            time.sleep(1.0)
        batch_count[0] += 1
        return run_ocr(batch)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} samples)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=None,
    ) as progress:
        task_id = progress.add_task(
            f"[cyan]Running OCR inference on {model_config.name}...",
            total=total_samples
        )
        
        progress_tracker["progress"] = progress
        progress_tracker["task_id"] = task_id
        
        dataset = dataset.map(
            run_ocr_with_delay, 
            batched=True, 
            batch_size=batch_size,
            load_from_cache_file=False 
        )
    
    raw_predictions = dataset["predictions"]
    
    if clean_predictions:
        logger.info("Cleaning predictions...")
        cleaned_predictions = [
            clean_ocr_text(pred) if pred else ""
            for pred in raw_predictions
        ]
    else:
        logger.info("Skipping prediction cleaning...")
        cleaned_predictions = [
            clean_html_markdown(pred) if pred else ""
            for pred in raw_predictions
        ]
    
    logger.info("Computing evaluation metrics...")
    
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    exact_match_metric = evaluate.load("exact_match")
    chrf_metric = evaluate.load("chrf")
    
    metrics_raw = {
        "cer": cer_metric.compute(predictions=raw_predictions, references=dataset[ground_truth_column]),
        "wer": wer_metric.compute(predictions=raw_predictions, references=dataset[ground_truth_column]),
        "exact_match": exact_match_metric.compute(predictions=raw_predictions, references=dataset[ground_truth_column]),
        "chrf": chrf_metric.compute(predictions=raw_predictions, references=dataset[ground_truth_column]),
        "bleu": bleu_metric.compute(predictions=raw_predictions, references=[[ref] for ref in dataset[ground_truth_column]]),
        "rouge": rouge_metric.compute(predictions=raw_predictions, references=dataset[ground_truth_column]),
        "meteor": meteor_metric.compute(predictions=raw_predictions, references=dataset[ground_truth_column]),
    }
    
    metrics_cleaned = {
        "cer": cer_metric.compute(predictions=cleaned_predictions, references=dataset[ground_truth_column]),
        "wer": wer_metric.compute(predictions=cleaned_predictions, references=dataset[ground_truth_column]),
        "exact_match": exact_match_metric.compute(predictions=cleaned_predictions, references=dataset[ground_truth_column]),
        "chrf": chrf_metric.compute(predictions=cleaned_predictions, references=dataset[ground_truth_column]),
        "bleu": bleu_metric.compute(predictions=cleaned_predictions, references=[[ref] for ref in dataset[ground_truth_column]]),
        "rouge": rouge_metric.compute(predictions=cleaned_predictions, references=dataset[ground_truth_column]),
        "meteor": meteor_metric.compute(predictions=cleaned_predictions, references=dataset[ground_truth_column]),
    }
    
    duration = time.time() - start_time
    
    detailed_results = []
    for i in range(len(dataset)):
        sample_result = {
            "sample_index": i,
            "predicted_text_raw": raw_predictions[i],
            "predicted_text_cleaned": cleaned_predictions[i],
            "reference_text": dataset[ground_truth_column][i],  
        }
        detailed_results.append(sample_result)
    
    dataset_info = {
        "dataset_name": dataset_name,
        "split": dataset_split,
    }
    
    if not dataset_name and hasattr(dataset, 'info') and dataset.info:
        dataset_info["dataset_name"] = getattr(dataset.info, 'dataset_name', None)
    if not dataset_split and hasattr(dataset, 'info') and dataset.info:
        dataset_info["split"] = getattr(dataset.info, 'split', None)
    
    results = {
        "model": model_config.name,
        "num_samples": len(dataset),
        "model_config": {
            "name": model_config.name,
            "url": model_config.get_url(),
            "prompt": model_config.prompt,
            "batch_size": model_config.batch_size,
            "max_workers": model_config.max_workers,
        },
        "dataset_info": {
            **dataset_info,
            "num_samples": len(dataset),
        },
        "metrics": {
            "raw": _serialize_metrics(metrics_raw),
            "cleaned": _serialize_metrics(metrics_cleaned),
        },
        "processing_config": {
            "batch_size": batch_size,
            "max_workers": max_workers,
        },
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "duration_formatted": f"{int(duration // 60)}m {int(duration % 60)}s",
        "samples": detailed_results,
    }
    
    if extra:
        results["extra"] = extra
    
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_config.name.replace('/', '_')}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        save_data = _make_serializable(results.copy())
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {filepath}")
    
    return results


def _serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serialized[key] = {k: float(v) if isinstance(v, (int, float, complex)) else str(v) 
                              for k, v in value.items()}
        elif isinstance(value, (int, float, complex)):
            serialized[key] = float(value)
        else:
            serialized[key] = str(value)
    return serialized


def _make_serializable(obj: Any) -> Any:
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

