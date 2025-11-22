import os
import json
import time
import logging
import evaluate
from datetime import datetime
from typing import Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..models.config import ModelEval, TranslationModel
from ..inference.ocr import infer
from ..inference.translation import infer as translate
from ..utils import (
    encode_image,
    extract_ground_truth_text,
    clean_html_markdown,
    clean_ocr_text,
)

logger = logging.getLogger(__name__)


def evaluate_dataset(
    dataset,
    model_config: Union[ModelEval, TranslationModel],
    task_type: str = "ocr",
    metric: Optional[Any] = None,
    image_column: str = "pixel_values",
    ground_truth_column: str = "text",
    source_column: str = "source",
    target_column: str = "target",
    ocr_column: Optional[str] = "ocr",
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
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
    Evaluate model on a dataset (OCR or Translation).

    For OCR: Computes core OCR metrics (CER, WER, chrF, Exact Match) and generation metrics (BLEU, ROUGE, METEOR) on both raw and cleaned predictions.
    For Translation: Computes translation metrics (BLEU, METEOR, chrF, ROUGE).

    Args:
        dataset: HuggingFace Dataset object
        model_config: ModelEval (for OCR) or TranslationModel (for translation) configuration instance
        task_type: Task type - "ocr" or "translation" (defaults to "ocr")
        metric: Optional single metric for backward compatibility (defaults to None, computes all metrics)
        image_column: Name of column containing images (OCR only)
        ground_truth_column: Name of column containing ground truth text (OCR only)
        source_column: Name of column containing source text (translation only)
        target_column: Name of column containing target/reference text (translation only)
        ocr_column: Name of column containing OCR data structure (OCR only, optional)
        src_lang: Source language code/name (translation only, overrides config default)
        target_lang: Target language code/name (translation only, overrides config default)
        batch_size: Number of samples per batch (OCR only, defaults to model_config.batch_size)
        max_workers: Number of parallel API calls (defaults to model_config.max_workers)
        num_samples: Number of samples to evaluate
        save_results: Whether to save results to JSON file
        results_dir: Directory to save results
        extra: Additional metadata to include in saved results
        clean_predictions: Whether to clean predictions (OCR only, defaults to True)
        dataset_name: Name of the dataset (optional, for tracking)
        dataset_split: Split name (e.g., 'train', 'test') (optional, for tracking)

    Returns:
        Dictionary containing evaluation results with metrics and samples
    """
    if task_type == "translation":
        if not isinstance(model_config, TranslationModel):
            raise ValueError(
                f"task_type='translation' requires TranslationModel, got {type(model_config).__name__}"
            )
        return _evaluate_translation(
            dataset=dataset,
            model_config=model_config,
            source_column=source_column,
            target_column=target_column,
            src_lang=src_lang,
            target_lang=target_lang,
            max_workers=max_workers,
            num_samples=num_samples,
            save_results=save_results,
            results_dir=results_dir,
            extra=extra,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
    else:
        if not isinstance(model_config, ModelEval):
            raise ValueError(
                f"task_type='ocr' requires ModelEval, got {type(model_config).__name__}"
            )
        return _evaluate_ocr(
            dataset=dataset,
            model_config=model_config,
            metric=metric,
            image_column=image_column,
            ground_truth_column=ground_truth_column,
            ocr_column=ocr_column,
            batch_size=batch_size,
            max_workers=max_workers,
            num_samples=num_samples,
            save_results=save_results,
            results_dir=results_dir,
            extra=extra,
            clean_predictions=clean_predictions,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )


def _evaluate_ocr(
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
    # Use model_config defaults if not provided
    batch_size = batch_size if batch_size is not None else model_config.batch_size
    max_workers = max_workers if max_workers is not None else model_config.max_workers
    num_samples = num_samples if num_samples is not None else model_config.num_samples
    save_results = (
        save_results if save_results is not None else model_config.save_results
    )
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
        raise ValueError(
            f"Ground truth column '{ground_truth_column}' not found in dataset"
        )

    logger.info(
        f"Processing {len(dataset)} samples with batch_size={batch_size}, max_workers={max_workers}..."
    )
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
                    img_rgb = (
                        Image.fromarray(img).convert("RGB")
                        if hasattr(img, "shape")
                        else img
                    )

                encoded_img = encode_image(img_rgb)
                return idx, infer(encoded_img, model_config, max_retries=3)
            except Exception as e:
                logger.warning(f"Error processing image {idx}: {type(e).__name__}: {e}")
                return idx, ""

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, idx, img): idx
                for idx, img in enumerate(images)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        batch["predictions"] = results

        if progress_tracker["progress"] and progress_tracker["task_id"] is not None:
            progress_tracker["completed"] += len(images)
            progress_tracker["progress"].update(
                progress_tracker["task_id"], completed=progress_tracker["completed"]
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
            total=total_samples,
        )

        progress_tracker["progress"] = progress
        progress_tracker["task_id"] = task_id

        dataset = dataset.map(
            run_ocr_with_delay,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=False,
        )

    raw_predictions = dataset["predictions"]

    if clean_predictions:
        logger.info("Cleaning predictions...")
        cleaned_predictions = [
            clean_ocr_text(pred) if pred else "" for pred in raw_predictions
        ]
    else:
        logger.info("Skipping prediction cleaning...")
        cleaned_predictions = [
            clean_html_markdown(pred) if pred else "" for pred in raw_predictions
        ]

    logger.info("Computing evaluation metrics...")

    # Core OCR metrics (standard for OCR evaluation)
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    exact_match_metric = evaluate.load("exact_match")
    chrf_metric = evaluate.load("chrf")

    # Generation metrics (for text generation quality assessment)
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    metrics_raw = {
        # Core OCR metrics
        "cer": cer_metric.compute(
            predictions=raw_predictions, references=dataset[ground_truth_column]
        ),
        "wer": wer_metric.compute(
            predictions=raw_predictions, references=dataset[ground_truth_column]
        ),
        "exact_match": exact_match_metric.compute(
            predictions=raw_predictions, references=dataset[ground_truth_column]
        ),
        "chrf": chrf_metric.compute(
            predictions=raw_predictions, references=dataset[ground_truth_column]
        ),
        # Generation metrics
        "bleu": bleu_metric.compute(
            predictions=raw_predictions,
            references=[[ref] for ref in dataset[ground_truth_column]],
        ),
        "rouge": rouge_metric.compute(
            predictions=raw_predictions, references=dataset[ground_truth_column]
        ),
        "meteor": meteor_metric.compute(
            predictions=raw_predictions, references=dataset[ground_truth_column]
        ),
    }

    metrics_cleaned = {
        # Core OCR metrics
        "cer": cer_metric.compute(
            predictions=cleaned_predictions, references=dataset[ground_truth_column]
        ),
        "wer": wer_metric.compute(
            predictions=cleaned_predictions, references=dataset[ground_truth_column]
        ),
        "exact_match": exact_match_metric.compute(
            predictions=cleaned_predictions, references=dataset[ground_truth_column]
        ),
        "chrf": chrf_metric.compute(
            predictions=cleaned_predictions, references=dataset[ground_truth_column]
        ),
        # Generation metrics
        "bleu": bleu_metric.compute(
            predictions=cleaned_predictions,
            references=[[ref] for ref in dataset[ground_truth_column]],
        ),
        "rouge": rouge_metric.compute(
            predictions=cleaned_predictions, references=dataset[ground_truth_column]
        ),
        "meteor": meteor_metric.compute(
            predictions=cleaned_predictions, references=dataset[ground_truth_column]
        ),
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

    if not dataset_name and hasattr(dataset, "info") and dataset.info:
        dataset_info["dataset_name"] = getattr(dataset.info, "dataset_name", None)
    if not dataset_split and hasattr(dataset, "info") and dataset.info:
        dataset_info["split"] = getattr(dataset.info, "split", None)

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
            serialized[key] = {
                k: float(v) if isinstance(v, (int, float, complex)) else str(v)
                for k, v in value.items()
            }
        elif isinstance(value, (int, float, complex)):
            serialized[key] = float(value)
        else:
            serialized[key] = str(value)
    return serialized


def _evaluate_translation(
    dataset,
    model_config: TranslationModel,
    source_column: str = "source",
    target_column: str = "target",
    src_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    max_workers: Optional[int] = None,
    num_samples: Optional[int] = None,
    save_results: Optional[bool] = None,
    results_dir: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    dataset_split: Optional[str] = None,
) -> Dict[str, Any]:
    max_workers = max_workers if max_workers is not None else model_config.max_workers
    num_samples = num_samples if num_samples is not None else None
    save_results = save_results if save_results is not None else True
    results_dir = results_dir if results_dir is not None else "results"
    extra = extra if extra is not None else model_config.extra

    src_lang = src_lang or model_config.src_lang
    target_lang = target_lang or model_config.target_lang

    if "translation" in dataset.column_names and (
        source_column not in dataset.column_names
        or target_column not in dataset.column_names
    ):
        if not src_lang or not target_lang:
            raise ValueError(
                f"Dataset has 'translation' field but source/target columns not found. "
                f"Source and target languages required (src_lang={src_lang}, target_lang={target_lang}). "
                f"Provide in config or as parameters."
            )

        def extract_translation(example):
            example["source"] = example["translation"][src_lang]
            example["target"] = example["translation"][target_lang]
            return example

        dataset = dataset.map(extract_translation)
        logger.info(
            f"Extracted source ({src_lang}) and target ({target_lang}) from translation field"
        )

    if source_column not in dataset.column_names:
        raise ValueError(f"Source column '{source_column}' not found in dataset")
    if target_column not in dataset.column_names:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    if not src_lang or not target_lang:
        raise ValueError(
            f"Source and target languages required. "
            f"Provide in config or as parameters."
        )

    logger.info(f"Processing {len(dataset)} samples with max_workers={max_workers}...")
    logger.info(f"Translation direction: {src_lang} -> {target_lang}")
    start_time = time.time()

    total_samples = len(dataset)

    progress_tracker = {"progress": None, "task_id": None, "completed": 0}

    def process_translation(idx, source_text):
        try:
            if not source_text or not source_text.strip():
                return idx, ""
            translation = translate(
                source_text, model_config, src_lang=src_lang, target_lang=target_lang
            )
            return idx, translation
        except Exception as e:
            logger.warning(f"Error translating sample {idx}: {type(e).__name__}: {e}")
            return idx, ""

    predictions = [""] * total_samples

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
            f"[cyan]Running translation inference on {model_config.name}...",
            total=total_samples,
        )

        progress_tracker["progress"] = progress
        progress_tracker["task_id"] = task_id

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_translation, idx, source_text): idx
                for idx, source_text in enumerate(dataset[source_column])
            }

            for future in as_completed(futures):
                idx, result = future.result()
                predictions[idx] = result
                progress_tracker["completed"] += 1
                progress_tracker["progress"].update(
                    progress_tracker["task_id"], completed=progress_tracker["completed"]
                )

    references = dataset[target_column]

    logger.info("Computing evaluation metrics...")

    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    chrf_metric = evaluate.load("chrf")
    ter_metric = evaluate.load("ter")

    metrics = {
        "bleu": bleu_metric.compute(
            predictions=predictions, references=[[ref] for ref in references]
        ),
        "meteor": meteor_metric.compute(predictions=predictions, references=references),
        "chrf": chrf_metric.compute(predictions=predictions, references=references),
        "ter": ter_metric.compute(predictions=predictions, references=references),
    }

    duration = time.time() - start_time

    detailed_results = []
    for i in range(len(dataset)):
        sample_result = {
            "sample_index": i,
            "source_text": dataset[source_column][i],
            "predicted_translation": predictions[i],
            "reference_translation": references[i],
        }
        detailed_results.append(sample_result)

    dataset_info = {
        "dataset_name": dataset_name,
        "split": dataset_split,
    }

    if not dataset_name and hasattr(dataset, "info") and dataset.info:
        dataset_info["dataset_name"] = getattr(dataset.info, "dataset_name", None)
    if not dataset_split and hasattr(dataset, "info") and dataset.info:
        dataset_info["split"] = getattr(dataset.info, "split", None)

    results = {
        "model": model_config.name,
        "model_type": model_config.type,
        "num_samples": len(dataset),
        "model_config": {
            "name": model_config.name,
            "type": model_config.type,
            "src_lang": src_lang,
            "target_lang": target_lang,
            "max_workers": model_config.max_workers,
        },
        "dataset_info": {
            **dataset_info,
            "num_samples": len(dataset),
        },
        "metrics": _serialize_metrics(metrics),
        "processing_config": {
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
