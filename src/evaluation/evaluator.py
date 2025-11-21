"""Dataset evaluation functions."""

import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from ..models.config import ModelEval
from ..inference.ocr import infer
from ..utils import encode_image, extract_ground_truth_text

logger = logging.getLogger(__name__)


def evaluate_dataset(
    dataset,
    metric,
    model_config: ModelEval,
    image_column: str = "pixel_values",
    ground_truth_column: str = "text",
    ocr_column: Optional[str] = "ocr",
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    num_samples: Optional[int] = None,
    save_results: Optional[bool] = None,
    results_dir: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate OCR model on a dataset.
    
    Args:
        dataset: HuggingFace Dataset object
        metric: Evaluation metric (e.g., evaluate.load("cer"))
        model_config: ModelEval configuration instance
        image_column: Name of column containing images
        ground_truth_column: Name of column containing ground truth text (will be created if ocr_column provided)
        ocr_column: Name of column containing OCR data structure (optional, used to extract ground truth)
        batch_size: Number of samples per batch (defaults to model_config.batch_size)
        max_workers: Number of parallel API calls (defaults to model_config.max_workers)
        num_samples: Number of samples to evaluate (defaults to model_config.num_samples)
        save_results: Whether to save results to JSON file (defaults to model_config.save_results)
        results_dir: Directory to save results (defaults to model_config.results_dir)
        extra: Additional metadata to include in saved results (defaults to model_config.extra)
    
    Returns:
        Dictionary containing metric results and predictions
    """
    # Use model_config defaults if not provided
    batch_size = batch_size if batch_size is not None else model_config.batch_size
    max_workers = max_workers if max_workers is not None else model_config.max_workers
    num_samples = num_samples if num_samples is not None else model_config.num_samples
    save_results = save_results if save_results is not None else model_config.save_results
    results_dir = results_dir if results_dir is not None else model_config.results_dir
    extra = extra if extra is not None else model_config.extra
    
    # Select subset if specified
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Extract ground truth text if ocr_column is provided
    # Also preserve bounding box information for detailed analysis
    if ocr_column and ocr_column in dataset.column_names:
        logger.info(f"Extracting ground truth text from '{ocr_column}' column...")
        def add_text_column(example):
            ocr_data = example.get(ocr_column, [])
            # Extract merged text for CER calculation
            example[ground_truth_column] = extract_ground_truth_text(ocr_data)
            # Preserve bounding boxes for detailed analysis
            if isinstance(ocr_data, list):
                example["bounding_boxes"] = ocr_data
            return example
        dataset = dataset.map(add_text_column)
    
    # Ensure ground truth column exists
    if ground_truth_column not in dataset.column_names:
        raise ValueError(f"Ground truth column '{ground_truth_column}' not found in dataset")
    
    # Process OCR on dataset
    logger.info(f"Processing {len(dataset)} samples with batch_size={batch_size}, max_workers={max_workers}...")
    start_time = time.time()
    
    def run_ocr(batch):
        """Run OCR on a batch of images with pipelined encoding and inference."""
        images = batch[image_column]
        results = [""] * len(images)
        
        def process_image(idx, img):
            """Encode and infer a single image."""
            try:
                # Convert to RGB
                if isinstance(img, Image.Image):
                    img_rgb = img.convert("RGB")
                else:
                    img_rgb = Image.fromarray(img).convert("RGB") if hasattr(img, 'shape') else img
                
                # Encode and infer in one go (with retry logic)
                encoded_img = encode_image(img_rgb)
                return idx, infer(encoded_img, model_config, max_retries=3)
            except Exception as e:
                logger.warning(f"Error processing image {idx}: {type(e).__name__}: {e}")
                return idx, ""
        
        # Process all images in parallel (encoding + inference)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_image, idx, img): idx for idx, img in enumerate(images)}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        batch["predictions"] = results
        return batch
    
    # Run OCR with rate limiting between batches
    logger.info(f"Processing {len(dataset)} samples in batches of {batch_size}...")
    
    # Process batches - dataset.map handles batching, we add delay in run_ocr
    batch_count = [0]  # Use list to allow modification in nested function
    
    def run_ocr_with_delay(batch):
        """Run OCR with delay between batches."""
        if batch_count[0] > 0:
            time.sleep(1.0)  # 1 second delay between batches to avoid overload
        batch_count[0] += 1
        return run_ocr(batch)
    
    dataset = dataset.map(run_ocr_with_delay, batched=True, batch_size=batch_size)
    
    # Compute metric
    logger.info("Computing evaluation metric...")
    metric_results = metric.compute(
        predictions=dataset["predictions"],
        references=dataset[ground_truth_column]
    )
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Build detailed results with bounding box information
    detailed_results = []
    for i in range(len(dataset)):
        sample_result = {
            "sample_index": i,
            "predicted_text": dataset["predictions"][i], 
            "reference_text": dataset[ground_truth_column][i],  
        }
        
        # Add bounding boxes with ground truth text (no predictions per box since model returns merged text)
        if "bounding_boxes" in dataset[i]:
            bounding_boxes = dataset[i]["bounding_boxes"]
            sample_result["bounding_boxes"] = []
            for bbox in bounding_boxes:
                bbox_entry = {
                    "bounding_box": bbox.get("bounding box", {}),
                    "ground_truth_text": bbox.get("text", ""),
                    # Note: No predicted_text per bbox - model returns single merged prediction
                }
                sample_result["bounding_boxes"].append(bbox_entry)
        
        detailed_results.append(sample_result)
    
    results = {
        "model": model_config.name,
        "num_samples": len(dataset),
        "batch_size": batch_size,
        "max_workers": max_workers,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "duration_formatted": f"{int(duration // 60)}m {int(duration % 60)}s",
        "metric": metric_results,
        "samples": detailed_results,
        # Keep flat lists for backward compatibility
        "predictions": dataset["predictions"],
        "references": dataset[ground_truth_column],
    }
    
    # Add extra metadata if provided
    if extra:
        results["extra"] = extra
    
    # Save results to JSON file
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_config.name.replace('/', '_')}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Convert metric_results to serializable format if needed
        save_data = results.copy()
        if isinstance(metric_results, dict):
            save_data["metric"] = {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                  for k, v in metric_results.items()}
        else:
            save_data["metric"] = float(metric_results) if isinstance(metric_results, (int, float)) else str(metric_results)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {filepath}")
    
    return results

