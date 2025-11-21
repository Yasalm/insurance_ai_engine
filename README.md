# Insurance AI Engine

A mono-repository for deploying and evaluating OCR (Optical Character Recognition) models for insurance document processing. This repository supports two distinct use cases:

1. **Model Deployment**: Deploy OCR models on GPU servers using vLLM for production inference
2. **Model Evaluation**: Run comprehensive evaluations on OCR models by calling deployed instances

## Overview

### Why Vision-Language Models?

Traditional OCR pipelines (Tesseract, PaddleOCR) struggle with:
- Complex layouts and multi-column documents
- Handwritten text mixed with printed text
- Table extraction and structured data
- Context-aware text recognition

Vision-language models (VLMs) address these limitations by:
- Understanding document structure through visual reasoning
- Extracting structured information (tables, forms) directly
- Handling diverse document types with a single model
- Providing natural language outputs that can be easily parsed

**Example Dataset Images:**

The evaluation dataset contains invoices with complex layouts, multi-column structures, and tabular data. The following example shows two invoice documents (English and French) that demonstrate the layout complexity models must handle:

![Dataset Example - Complex Invoice Layouts](docs/images/output.png)

These documents require models capable of:
- Extracting structured table data (itemized lists, quantities, prices)
- Preserving spatial relationships between text elements
- Handling multi-column layouts
- Recognizing document structure (headers, footers, tables, signatures)

### Selected Models

The repository currently evaluates vision-language models specifically designed for OCR tasks:

**nanonets/Nanonets-OCR2-3B**
- 3B parameter vision-language model optimized for document OCR
- Capable of extracting text from complex layouts including tables
- Outputs structured formats (HTML tables, markdown) when appropriate
- Currently active and evaluated

**rednote-hilab/dots.ocr**
- Specialized OCR model for document processing
- Designed for structured document extraction
- Currently configured but inactive (can be activated for comparison)

**Selection Criteria:**
- **Layout Awareness**: Handle documents with tables and structured layouts (invoices, forms, receipts)
- **Text Extraction Quality**: Accurate text extraction for downstream processing
- **API Compatibility**: Deployable via vLLM with OpenAI-compatible API
- **Production Readiness**: Suitable for production with reasonable inference speed

### Evaluation Approach

The evaluation uses text-based metrics to assess model performance on insurance documents with structured layouts and tables.

**Evaluation Methods:**

1. **Core OCR Metrics**: Uses standard OCR metrics (CER, WER, chrF, Exact Match) to measure content extraction accuracy, not layout preservation.

2. **Generation Metrics**: Includes BLEU, ROUGE, METEOR to assess text generation quality, since these are generative vision-language models.

3. **Raw vs Cleaned Comparison**: Evaluates both raw and cleaned outputs because evaluating layout is complex. Cleaning ensures fair comparison with ground truth text that doesn't include markup.

4. **Multiple Metrics**: Different metrics measure different aspects:
   - **Core OCR (Character-level)**: CER, chrF
   - **Core OCR (Word-level)**: WER, Exact Match
   - **Generation (Sequence-level)**: BLEU, ROUGE, METEOR

## Architecture

This is a mono-repo that can be cloned and used in two ways:

- **GPU Server Deployment**: Clone on a GPU server and serve OCR models via vLLM with an OpenAI-compatible API
- **AI Engineer Evaluation**: Clone locally and run evaluations by calling the deployed model instances remotely

The architecture separates concerns: model serving happens on GPU infrastructure, while evaluation can be done from any machine that can reach the deployed API.

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Server (Deployment)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  vLLM Server (OpenAI-compatible API)                  │  │
│  │  - nanonets/Nanonets-OCR2-3B                         │  │
│  │  - rednote-hilab/dots.ocr                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ▲                                  │
│                          │ HTTP API Calls                   │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    Local Machine (Evaluation)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Evaluation Scripts                                  │  │
│  │  - Dataset loading                                   │  │
│  │  - OCR inference (via API)                          │  │
│  │  - Metric computation                               │  │
│  │  - Results visualization                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

Model configuration is managed through `src/config/models.yaml`. Each model entry includes:

- `name`: Model identifier (e.g., `nanonets/Nanonets-OCR2-3B`)
- `prompt`: Instruction prompt for the OCR task
- `batch_size`: Number of samples processed per batch during evaluation
- `max_workers`: Maximum parallel API calls for batch inference
- `num_samples`: Number of samples to evaluate (optional)
- `active`: Whether the model is active (only active models are loaded)
- `save_results`: Whether to save evaluation results to JSON
- `results_dir`: Directory for saving results
- `url`: Optional explicit API URL (can also be set via environment variables)

### Environment Variables

Model URLs can be configured via environment variables:

- `{MODEL_NAME}_URL`: Specific model URL (e.g., `NANONETS_NANONETS_OCR2_3B_URL`)
- `MODEL_URL`: Fallback URL for all models

Example:
```bash
export NANONETS_NANONETS_OCR2_3B_URL="http://localhost:8002/v1"
export OPENAI_API_KEY="DUMMY_API_KEY"  # Required but can be dummy for local vLLM
```

## Deployment Guidelines

### Prerequisites

- GPU server with CUDA support
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Sufficient GPU memory (models require 3B+ parameters)

### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd insurance_ai_engine
```

### Step 2: Install Dependencies

```bash
# Option 1: Use the makefile helper (recommended)
make setup-uv-and-sync

# Option 2: Manual setup
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### Step 3: Deploy Model Server

The repository includes Makefile targets for deploying models:

```bash
# Deploy Nanonets OCR2 3B model (default port 8002)
make ocr-serve-nanonets [PORT=8002]

# Deploy Dots OCR model
make ocr-serve-dots [PORT=8002]

# Stop all vLLM servers
make stop
```

### Step 4: Verify Deployment

Check that the server is running:

```bash
curl http://localhost:8002/health
```

The vLLM server exposes an OpenAI-compatible API at `http://localhost:8002/v1`. You can test it:

```bash
curl http://localhost:8002/v1/models
```

### Production Considerations

- **GPU Memory**: Adjust `--gpu-memory-utilization` based on available VRAM
- **Concurrency**: Tune `--max-num-seqs` and `--max-num-batched-tokens` for your workload
- **Prefix Caching**: Enabled by default for better performance with repeated prompts
- **Async Scheduling**: Recommended for handling multiple concurrent requests
- **Port Configuration**: Use environment variables or Makefile PORT parameter
- **Process Management**: Consider using systemd or supervisor for production deployments

## Evaluation Setup

### Prerequisites

- Python 3.12+
- Access to deployed model API (local or remote)
- HuggingFace datasets access (for downloading evaluation datasets)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd insurance_ai_engine

# Setup dependencies
make setup-uv-and-sync
```

### Step 2: Configure Model URL

Set the environment variable pointing to your deployed model:

```bash
export NANONETS_NANONETS_OCR2_3B_URL="http://your-gpu-server:8002/v1"
export OPENAI_API_KEY="DUMMY_API_KEY"
```

### Step 3: Run Evaluation

```bash
# Run evaluation on all active models
make run-evaluate

# Or run directly
uv run python -m src.scripts.evaluate
```

The evaluation script loads the dataset, runs OCR inference via API calls, computes metrics on raw and cleaned predictions, displays results in a table, and saves detailed JSON results.

### Evaluation Results JSON

Each evaluation saves a JSON file to `results/` with pattern: `{model_name}_{timestamp}.json`

```json
{
  "model": "nanonets/Nanonets-OCR2-3B",           // Model identifier
  "num_samples": 100,                              // Number of samples evaluated
  "timestamp": "2025-11-21T19:07:28.364252",      // ISO format timestamp
  "duration_seconds": 395.35,                     // Total evaluation time
  "duration_formatted": "6m 35s",                 // Human-readable duration
  
  "model_config": {                                // Model configuration used
    "name": "nanonets/Nanonets-OCR2-3B",
    "url": "http://localhost:8002/v1",
    "prompt": "Extract the text from the above document...",
    "batch_size": 10,
    "max_workers": 3
  },
  
  "dataset_info": {                                // Dataset metadata
    "dataset_name": "amaye15/invoices-google-ocr",
    "split": "test",
    "num_samples": 100
  },
  
  "metrics": {                                     // Evaluation metrics
    "raw": {                                       // Raw model output (with HTML/markdown)
      // Core OCR metrics
      "cer": 0.87,                                 // Character Error Rate
      "wer": 0.84,                                 // Word Error Rate
      "exact_match": { "exact_match": 0.0 },      // Exact Match percentage
      "chrf": { "score": 42.19, ... },            // Character n-gram F-score
      // Generation metrics
      "bleu": { "bleu": 0.19, ... },              // BLEU score
      "rouge": { "rouge1": 0.56, "rouge2": 0.44, "rougeL": 0.52 }, // ROUGE scores
      "meteor": { "meteor": 0.42 }                // METEOR score
    },
    "cleaned": {                                   // Cleaned output (HTML/markdown removed)
      // Core OCR metrics
      "cer": 0.80,
      "wer": 1.03,
      "exact_match": { "exact_match": 0.0 },
      "chrf": { "score": 45.84, ... },
      // Generation metrics
      "bleu": { "bleu": 0.28, ... },
      "rouge": { "rouge1": 0.58, "rouge2": 0.51, "rougeL": 0.55 },
      "meteor": { "meteor": 0.41 }
    }
  },
  
  "processing_config": {                           // Processing parameters
    "batch_size": 10,
    "max_workers": 3
  },
  
  "samples": [                                     // Per-sample detailed results
    {
      "sample_index": 0,
      "predicted_text_raw": "...",                 // Raw model output
      "predicted_text_cleaned": "...",             // Cleaned output
      "reference_text": "..."                      // Ground truth
    }
  ],
  
  "extra": {                                       // Optional metadata
    "gpu": "L40"
  }
}
```

Example output from the evaluation script:

![Evaluation Results Table](docs/images/evaluation_results_table.png)

The table shows evaluation metrics comparing raw and cleaned predictions. Key observations:
- **Core OCR metrics**: CER improves from 0.8703 (raw) to 0.7962 (cleaned), WER shows the impact of cleaning on word-level accuracy
- **Generation metrics**: BLEU, ROUGE, and METEOR demonstrate improvements from text normalization, assessing text generation quality

## Evaluation Metrics

The evaluation uses two categories of metrics:

**Core OCR Metrics** (standard for OCR evaluation):
- **CER** (Character Error Rate): Primary OCR metric for character-level accuracy
- **WER** (Word Error Rate): Primary OCR metric for word-level accuracy  
- **chrF** (Character n-gram F-score): Character-level similarity with order awareness
- **Exact Match**: Percentage of perfectly matched samples

**Generation Metrics** (for text generation quality assessment):
- **BLEU**: N-gram precision for text generation quality
- **ROUGE**: Recall-oriented metrics for text generation quality
- **METEOR**: Semantic similarity with synonym matching

| Metric | Type | Category | Description | Direction | Use Case |
|--------|------|----------|-------------|-----------|----------|
| **CER** (Character Error Rate) | Character | Core OCR | Percentage of characters that differ between prediction and ground truth. Formula: `(Substitutions + Insertions + Deletions) / Total Characters` | Lower is better (0.0 = perfect) | Primary OCR metric for character-level accuracy |
| **WER** (Word Error Rate) | Word | Core OCR | Percentage of words that differ between prediction and ground truth. Formula: `(Substitutions + Insertions + Deletions) / Total Words` | Lower is better (0.0 = perfect) | Primary OCR metric for word-level accuracy |
| **chrF** (Character n-gram F-score) | Character | Core OCR | Harmonic mean of character precision and recall using n-grams. Considers character sequences, not just individual characters | Higher is better (0-100 scale, normalized to 0-1) | Character-level similarity with order awareness |
| **Exact Match** | Word | Core OCR | Percentage of samples where the entire prediction exactly matches ground truth. Very strict metric - any difference fails | Higher is better (0.0-1.0) | Assessing perfect accuracy rate |
| **BLEU** | Sequence | Generation | Measures n-gram precision between prediction and reference. Originally designed for machine translation, used here for text generation quality | Higher is better (0.0-1.0) | Text generation quality assessment |
| **ROUGE-1** | Sequence | Generation | Unigram recall - how many words from reference appear in prediction | Higher is better (0.0-1.0) | Text generation quality assessment |
| **ROUGE-2** | Sequence | Generation | Bigram recall - how many word pairs match | Higher is better (0.0-1.0) | Text generation quality assessment |
| **ROUGE-L** | Sequence | Generation | Longest Common Subsequence - captures sentence structure | Higher is better (0.0-1.0) | Text generation quality assessment |
| **METEOR** | Sequence | Generation | Harmonic mean of precision and recall with synonym matching. Considers word order and semantic similarity | Higher is better (0.0-1.0) | Text generation quality assessment |

### Limitations and Future Enhancements

**Current Limitations:**

The evaluation focuses on text-based metrics only. While models can extract structured layouts (tables, HTML), current metrics don't assess layout preservation, HTML/XML structure accuracy, spatial relationships, or table-specific metrics.

**Future Enhancements:**

Since models output structured formats (HTML tables, markdown), future enhancements could include layout evaluation metrics, HTML/XML parsing accuracy, spatial relationship metrics, and table-specific evaluation.


### Evaluation Results Visualization

Results from running text-based metrics on 100 invoice samples, comparing raw VLM output and cleaned versions:

![Evaluation Charts](docs/images/evaluation_charts.png)

## Dataset and Text Cleaning

### Dataset Structure

The evaluation uses the `amaye15/invoices-google-ocr` dataset, which contains:

- **Images**: Invoice and document images (PNG format)
- **OCR Annotations**: Structured OCR data with bounding boxes and text
- **Labels**: Document type classification (Invoice, Receipt, Barcode, etc.). Note: Only invoice samples were used in the current evaluation

The dataset includes documents with tables and structured layouts, which is why models capable of layout extraction are preferred. However, the current evaluation focuses on text-based metrics rather than layout preservation.

### Cleaning Process

The evaluation pipeline includes two cleaning functions:

**`clean_html_markdown()`**: Removes HTML tags and markdown syntax
- Strips `<table>`, `<tr>`, `<td>` tags
- Removes markdown headers, lists, code blocks
- Preserves text content

**`clean_ocr_text()`**: Advanced OCR-specific normalization
- Unicode normalization (smart quotes, dashes, etc.)
- Whitespace normalization
- Punctuation spacing fixes
- Zero-width character removal

This cleaning ensures that:
- Models that intentionally output HTML tables and markdown are evaluated fairly on text content
- Text extraction accuracy is measured separately from layout preservation
- Metrics reflect OCR accuracy, not differences in markup formatting

## API Server

The repository includes a FastAPI server (`server/main.py`) that provides a REST API for OCR inference. This can be deployed alongside or separately from the vLLM model server.

### Running the API Server

```bash
# Development
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

- `GET /health` - Health check
- `GET /models` - List active models
- `POST /ocr/infer` - Single image/PDF inference
- `POST /ocr/infer-base64` - Base64 image inference
- `POST /ocr/batch` - Batch inference

See `server/README.md` for detailed API documentation.

## Project Structure

```
insurance_ai_engine/
├── src/
│   ├── config/           # Configuration loading (models.yaml)
│   ├── evaluation/       # Evaluation metrics and logic
│   ├── inference/        # OCR inference via API
│   ├── models/           # Model configuration models
│   ├── scripts/           # Evaluation scripts
│   │   ├── evaluate.py   # Main evaluation script
│   │   └── data_loader.py # Dataset loading utilities
│   └── utils.py          # Utility functions (image encoding, text cleaning)
├── server/               # FastAPI server for OCR API
├── notebooks/            # Jupyter notebooks for experimentation
│   └── 01_experminets.ipynb # Evaluation experiments and visualizations
├── results/              # Evaluation results (JSON files)
├── data/                 # Dataset cache directory
├── makefile              # Deployment and utility commands
└── pyproject.toml        # Python dependencies
```

## Model Comparison

To compare multiple models:

1. Configure multiple models in `src/config/models.yaml` with `active: true`
2. Set appropriate URLs for each model via environment variables
3. Run `make run-evaluate` - it evaluates all active models

Results are saved to separate timestamped JSON files in `results/` for comparison.

## Usage Examples

### Deploy Model on GPU Server

```bash
# On GPU server
make ocr-serve-nanonets PORT=8002
```

### Run Evaluation from Local Machine

```bash
# On local machine
export NANONETS_NANONETS_OCR2_3B_URL="http://gpu-server:8002/v1"
make run-evaluate
```

### Compare Multiple Models

```bash
# Set URLs for multiple models
export NANONETS_NANONETS_OCR2_3B_URL="http://gpu-server-1:8002/v1"
export REDNOTE_HILAB_DOTS_OCR_URL="http://gpu-server-2:8003/v1"

# Run evaluation (will evaluate all active models)
make run-evaluate

# Compare results from JSON files
ls -lh results/
```

### Custom Evaluation

You can also use the evaluation framework programmatically:

```python
from datasets import load_dataset
from src.config import load_config
from src.evaluation import evaluate_dataset
from src.scripts.data_loader import download_dataset

# Load configuration
config = load_config()

# Option 1: Load dataset directly
dataset = load_dataset("amaye15/invoices-google-ocr", split="test")

# Option 2: Use the data loader utility
dataset = download_dataset(
    dataset_name="amaye15/invoices-google-ocr",
    split="test",
    cache_dir="data"
)

# Run evaluation
results = evaluate_dataset(
    dataset=dataset,
    model_config=config.ocr_models[0],
    num_samples=100,
    dataset_name="amaye15/invoices-google-ocr",
    dataset_split="test"
)

# Access metrics
# Core OCR metrics
print(f"CER (cleaned): {results['metrics']['cleaned']['cer']}")
print(f"WER (cleaned): {results['metrics']['cleaned']['wer']}")
print(f"chrF (cleaned): {results['metrics']['cleaned']['chrf']['score']}")
print(f"Exact Match (cleaned): {results['metrics']['cleaned']['exact_match']['exact_match']}")
# Generation metrics
print(f"BLEU (cleaned): {results['metrics']['cleaned']['bleu']['bleu']}")
print(f"ROUGE-1 (cleaned): {results['metrics']['cleaned']['rouge']['rouge1']}")
```

### Notebook Experiments

The `notebooks/01_experminets.ipynb` notebook contains:
- Dataset exploration and analysis
- Custom evaluation runs
- Visualization code for generating comparison charts
- Experimentation with different model configurations

You can use this notebook as a reference for custom evaluations and visualizations.

## Development

### Adding a New Model

1. Add model configuration to `src/config/models.yaml`:

```yaml
ocr_models:
  - name: your-model/name
    prompt: "Your OCR prompt"
    batch_size: 10
    max_workers: 3
    num_samples: 100
    active: true
    save_results: true
    results_dir: results
```

2. Add deployment command to `makefile`:

```makefile
ocr-serve-your-model:
    uv run vllm serve your-model/name --host 0.0.0.0 --port $(PORT) ...
```

3. Set environment variable for model URL:

```bash
export YOUR_MODEL_NAME_URL="http://localhost:8002/v1"
```

### Available Makefile Commands

View all available commands:

```bash
make help
```

Common commands:
- `make setup-uv-and-sync` - Setup uv and install dependencies
- `make ocr-serve-nanonets` - Deploy Nanonets OCR model
- `make ocr-serve-dots` - Deploy Dots OCR model
- `make run-evaluate` - Run evaluation on active models
- `make stop` - Stop all vLLM servers
- `make clean` - Clean project artifacts

### Running Tests

```bash
# Run evaluation script
make run-evaluate
```

## Troubleshooting

### Model Server Not Starting

- Check GPU availability: `nvidia-smi`
- Verify CUDA installation
- Check port availability: `lsof -i :8002`
- Review vLLM logs for memory errors

### Evaluation Failing

- Verify model URL is accessible: `curl $MODEL_URL/health`
- Check environment variables are set correctly
- Ensure dataset can be downloaded (HuggingFace access)
- Review API timeout settings in `src/inference/ocr.py`

### High Memory Usage

- Reduce `batch_size` in model configuration
- Lower `max_workers` for parallel API calls
- Use smaller `num_samples` for testing
