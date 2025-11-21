"""Script to run OCR evaluation on datasets."""

from datasets import load_dataset
import evaluate
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
    """Run OCR evaluation on the invoices dataset."""
    # Load config
    config = load_config()
    
    if not config.ocr_models:
        logger.error("No active models found in config!")
        return
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "amaye15/invoices-google-ocr", split="test", cache_dir="../../data"
    )

    # Load evaluation metric
    logger.info("Loading CER metric...")
    cer = evaluate.load("cer")

    # Run evaluation for each active model
    for model_config in config.ocr_models:
        logger.info(f"Evaluating model: {model_config.name}")
        results = evaluate_dataset(
            dataset=dataset, 
            metric=cer,
            model_config=model_config
        )

        console.print()
        table = Table(
            title=f"Evaluation Results - {model_config.name}", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Number of samples", str(results["num_samples"]))

        # Format metric results
        if isinstance(results["metric"], dict):
            for key, value in results["metric"].items():
                table.add_row(
                    key, f"{value:.4f}" if isinstance(value, float) else str(value)
                )
        else:
            table.add_row(
                "CER Score",
                (
                    f"{results['metric']:.4f}"
                    if isinstance(results["metric"], float)
                    else str(results["metric"])
                ),
            )

        console.print(table)
        console.print()


if __name__ == "__main__":
    main()

