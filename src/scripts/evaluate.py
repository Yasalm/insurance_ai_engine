from datasets import load_dataset
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from src.evaluation import evaluate_dataset
from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


def main():
    config = load_config()
    
    if not config.ocr_models:
        logger.error("No active models found in config!")
        return
    
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "amaye15/invoices-google-ocr", split="test", cache_dir="../../data"
    )

    for model_config in config.ocr_models:
        logger.info(f"Evaluating model: {model_config.name}")
        results = evaluate_dataset(
            dataset=dataset,
            model_config=model_config,
            dataset_name="amaye15/invoices-google-ocr",
            dataset_split="test",
            num_samples=100,
            extra={
                "gpu": "L40",
            },
        )

        console.print()
        table = Table(
            title=f"Evaluation Results - {model_config.name}", 
            show_header=True, 
            header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Raw", style="red")
        table.add_column("Cleaned", style="green")

        table.add_row("Number of samples", str(results["dataset_info"]["num_samples"]), "")

        metrics_raw = results["metrics"]["raw"]
        metrics_cleaned = results["metrics"]["cleaned"]

        table.add_row("CER", f"{metrics_raw['cer']:.4f}", f"{metrics_cleaned['cer']:.4f}")
        table.add_row("WER", f"{metrics_raw['wer']:.4f}", f"{metrics_cleaned['wer']:.4f}")
        table.add_row("Exact Match", f"{metrics_raw['exact_match']['exact_match']:.4f}", f"{metrics_cleaned['exact_match']['exact_match']:.4f}")
        table.add_row("chrF", f"{metrics_raw['chrf']['score']:.4f}", f"{metrics_cleaned['chrf']['score']:.4f}")
        table.add_row("BLEU", f"{metrics_raw['bleu']['bleu']:.4f}", f"{metrics_cleaned['bleu']['bleu']:.4f}")
        table.add_row("ROUGE-1", f"{metrics_raw['rouge']['rouge1']:.4f}", f"{metrics_cleaned['rouge']['rouge1']:.4f}")
        table.add_row("ROUGE-2", f"{metrics_raw['rouge']['rouge2']:.4f}", f"{metrics_cleaned['rouge']['rouge2']:.4f}")
        table.add_row("ROUGE-L", f"{metrics_raw['rouge']['rougeL']:.4f}", f"{metrics_cleaned['rouge']['rougeL']:.4f}")
        table.add_row("METEOR", f"{metrics_raw['meteor']['meteor']:.4f}", f"{metrics_cleaned['meteor']['meteor']:.4f}")

        console.print(table)
        console.print()


if __name__ == "__main__":
    main()
