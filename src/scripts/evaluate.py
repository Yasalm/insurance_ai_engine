import argparse
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from src.evaluation import evaluate_dataset
from src.config import load_config
from src.scripts.data_loader import download_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def evaluate_ocr(config, dataset_name, dataset_split, num_samples, extra):
    """Evaluate OCR models."""
    if not config.ocr_models:
        logger.error("No active OCR models found in config!")
        return

    logger.info(f"Loading dataset: {dataset_name} (split: {dataset_split})")
    dataset = download_dataset(dataset_name=dataset_name, split=dataset_split)

    for model_config in config.ocr_models:
        logger.info(f"Evaluating OCR model: {model_config.name}")
        results = evaluate_dataset(
            dataset=dataset,
            model_config=model_config,
            task_type="ocr",
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            num_samples=num_samples,
            extra=extra,
        )

        console.print()
        table = Table(
            title=f"OCR Evaluation Results - {model_config.name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Raw", style="red")
        table.add_column("Cleaned", style="green")

        table.add_row("Number of samples", str(results["dataset_info"]["num_samples"]), "")

        metrics_raw = results["metrics"]["raw"]
        metrics_cleaned = results["metrics"]["cleaned"]

        table.add_row("CER", f"{metrics_raw['cer']:.4f}", f"{metrics_cleaned['cer']:.4f}")
        table.add_row("WER", f"{metrics_raw['wer']:.4f}", f"{metrics_cleaned['wer']:.4f}")
        table.add_row(
            "Exact Match",
            f"{metrics_raw['exact_match']['exact_match']:.4f}",
            f"{metrics_cleaned['exact_match']['exact_match']:.4f}",
        )
        table.add_row("chrF", f"{metrics_raw['chrf']['score']:.4f}", f"{metrics_cleaned['chrf']['score']:.4f}")
        table.add_row("BLEU", f"{metrics_raw['bleu']['bleu']:.4f}", f"{metrics_cleaned['bleu']['bleu']:.4f}")
        table.add_row(
            "ROUGE-1", f"{metrics_raw['rouge']['rouge1']:.4f}", f"{metrics_cleaned['rouge']['rouge1']:.4f}"
        )
        table.add_row(
            "ROUGE-2", f"{metrics_raw['rouge']['rouge2']:.4f}", f"{metrics_cleaned['rouge']['rouge2']:.4f}"
        )
        table.add_row(
            "ROUGE-L", f"{metrics_raw['rouge']['rougeL']:.4f}", f"{metrics_cleaned['rouge']['rougeL']:.4f}"
        )
        table.add_row("METEOR", f"{metrics_raw['meteor']['meteor']:.4f}", f"{metrics_cleaned['meteor']['meteor']:.4f}")

        console.print(table)
        console.print()


def evaluate_translation(config, dataset_name, dataset_config, dataset_split, num_samples, extra, src_lang=None, tgt_lang=None):
    """Evaluate translation models."""
    if not config.translation_models:
        logger.error("No active translation models found in config!")
        return

    logger.info(f"Loading translation dataset: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
    dataset = download_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split
    )

    if "translation" in dataset.column_names:
        if dataset_config and "-" in dataset_config:
            lang_pair = dataset_config.split("-")
            src_lang_code = lang_pair[0]
            tgt_lang_code = lang_pair[1]
        elif src_lang and tgt_lang:
            src_lang_code = src_lang
            tgt_lang_code = tgt_lang
        else:
            logger.error("Cannot determine language pair. Provide --dataset-config (e.g., 'en-fr') or --src-lang and --tgt-lang")
            return

        def extract_translation(example):
            example["source"] = example["translation"][src_lang_code]
            example["target"] = example["translation"][tgt_lang_code]
            return example

        dataset = dataset.map(extract_translation)
        logger.info(f"Extracted source ({src_lang_code}) and target ({tgt_lang_code}) from translation field")
    elif "source" in dataset.column_names and "target" in dataset.column_names:
        logger.info("Dataset already has 'source' and 'target' columns")
    else:
        logger.error("Dataset must have either 'translation' field or 'source'/'target' columns")
        return

    for model_config in config.translation_models:
        logger.info(f"Evaluating translation model: {model_config.name}")
        results = evaluate_dataset(
            dataset=dataset,
            model_config=model_config,
            task_type="translation",
            source_column="source",
            target_column="target",
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            num_samples=num_samples,
            extra=extra,
        )

        console.print()
        table = Table(
            title=f"Translation Evaluation Results - {model_config.name}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")

        table.add_row("Number of samples", str(results["dataset_info"]["num_samples"]))
        table.add_row("Source Language", results["model_config"]["src_lang"])
        table.add_row("Target Language", results["model_config"]["target_lang"])

        metrics = results["metrics"]

        table.add_row("BLEU", f"{metrics['bleu']['bleu']:.4f}")
        table.add_row("METEOR", f"{metrics['meteor']['meteor']:.4f}")
        table.add_row("chrF", f"{metrics['chrf']['score']:.4f}")
        table.add_row("ROUGE-1", f"{metrics['rouge']['rouge1']:.4f}")
        table.add_row("ROUGE-2", f"{metrics['rouge']['rouge2']:.4f}")
        table.add_row("ROUGE-L", f"{metrics['rouge']['rougeL']:.4f}")

        console.print(table)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR or Translation models")
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["ocr", "translation"],
        help="Task type: 'ocr' or 'translation'",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., 'amaye15/invoices-google-ocr' for OCR, 'Helsinki-NLP/opus-100' or 'wmt14' for translation)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset config (e.g., 'en-fr' for OPUS-100 or 'de-en' for WMT14 translation dataset)",
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default=None,
        help="Source language code (e.g., 'en', 'de'). Used when dataset has 'translation' field.",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default=None,
        help="Target language code (e.g., 'fr', 'en'). Used when dataset has 'translation' field.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: 'test')")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate (default: 100)")

    args = parser.parse_args()

    config = load_config()

    if args.task_type:
        task_type = args.task_type
    else:
        if config.ocr_models and not config.translation_models:
            task_type = "ocr"
        elif config.translation_models and not config.ocr_models:
            task_type = "translation"
        else:
            logger.error(
                "Both OCR and translation models found. Please specify --task-type (ocr or translation)"
            )
            return


    if task_type == "ocr":
        dataset_name = args.dataset or "amaye15/invoices-google-ocr"
        extra = {"gpu": "L40"}
        evaluate_ocr(config, dataset_name, args.split, args.num_samples, extra)
    elif task_type == "translation":
        dataset_name = args.dataset or "Helsinki-NLP/opus-100"
        dataset_config = args.dataset_config or "en-fr"
        evaluate_translation(
            config,
            dataset_name,
            dataset_config,
            args.split,
            args.num_samples,
            extra,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )


if __name__ == "__main__":
    main()
