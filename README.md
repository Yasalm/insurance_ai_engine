# Insurance AI Engine

A mono-repository for deploying, evaluating, and serving OCR and Translation models for insurance document processing. This system provides end-to-end capabilities from model deployment to web-based inference.

## Table of Contents

- [Assumptions and Prerequisites](#assumptions-and-prerequisites)
- [Overview](#overview)
- [OCR System](#ocr-system)
- [Translation System](#translation-system)
- [Evaluation Framework](#evaluation-framework)
- [System Architecture](#system-architecture)
- [Components](#components)
  - [API Server](#api-server)
  - [Web Interface](#web-interface)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Assumptions and Prerequisites

Before using this repository, please ensure you understand and have access to the following:

### Assumptions

1. **GPU Infrastructure**: You have access to GPU servers (local or cloud) capable of running large language models (3B+ parameters)
2. **Model Deployment**: OCR models are deployed via vLLM with OpenAI-compatible APIs
3. **Network Access**: Your deployment environment allows HTTP/HTTPS API calls between components
4. **Python Environment**: Python 3.12+ is available with package management via `uv`
5. **HuggingFace Access**: For downloading evaluation datasets (if running evaluations) no account needed

### Prerequisites

**For Model Deployment:**
- GPU server with CUDA support (A100, RTX 4090, or similar)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Sufficient GPU memory (models require 3B+ parameters, ~8GB+ VRAM recommended)

**For API Server & Web Interface:**
- Python 3.12+
- Access to deployed OCR and translation models (local or remote)

**For Evaluation:**
- Python 3.12+
- Access to deployed model APIs
- HuggingFace access

---

## Overview

### What is This Repository?

This repository provides a complete system for:

1. **OCR Model Deployment**: Deploy vision-language models (VLMs) for document OCR using vLLM
2. **Translation Services**: Deploy and use mBART models for multilingual text translation
3. **REST API Server**: FastAPI-based server exposing OCR and translation endpoints
4. **Web Interface**: Simple web UI for testing and demonstrating capabilities
5. **Model Evaluation**: Evaluation framework with multiple metrics

### Why Vision-Language Models for OCR?

Traditional OCR pipelines (Tesseract, PaddleOCR) struggle with:
- Complex layouts and multi-column documents
- Handwritten text mixed with printed text
- Table extraction and structured data
- Context-aware text recognition

**Vision-Language Models (VLMs) address these limitations by:**
- Understanding document structure through visual reasoning
- Extracting structured information (tables, forms) directly
- Handling diverse document types with a single model
- Providing natural language outputs that can be easily parsed

**Example Dataset Complexity:**

![Dataset Example - Complex Invoice Layouts](docs/images/output.png)

The evaluation dataset contains invoices with complex layouts, multi-column structures, and tabular data. These documents require models capable of:
- Extracting structured table data (itemized lists, quantities, prices)
- Preserving spatial relationships between text elements
- Handling multi-column layouts
- Recognizing document structure (headers, footers, tables, signatures)

---

## OCR System

The OCR system uses vision-language models to extract text from documents.

### Supported Models

**nanonets/Nanonets-OCR2-3B**
- 3B parameter vision-language model optimized for document OCR
- Capable of extracting text from complex layouts including tables
- Outputs structured formats (HTML tables, markdown) when appropriate
- Currently active and evaluated

**rednote-hilab/dots.ocr**
- Specialized OCR model for document processing
- Designed for structured document extraction
- Currently configured but inactive (can be activated for comparison)

### Model Selection Criteria

- **Layout Awareness**: Handle documents with tables and structured layouts
- **Text Extraction Quality**: Accurate text extraction for downstream processing
- **API Compatibility**: Deployable via vLLM with OpenAI-compatible API
- **Production Readiness**: Suitable for production with reasonable inference speed

### OCR Capabilities

- **Document Types**: Images (PNG, JPEG, GIF, BMP, WebP) and PDF files
- **Output Formats**: Plain text, HTML tables, Markdown
- **Batch Processing**: Process multiple documents in parallel
- **Structured Extraction**: Automatically extracts tables and structured data

---

## Translation System

The translation system provides multilingual text translation using mBART models.

### Supported Model

**facebook/mbart-large-50-many-to-many-mmt**
- Large multilingual translation model (611M parameters)
- Supports 50+ languages with proper language code prefixes
- Many-to-many translation architecture (any language to any language)

### Why mBART?

**Architecture Choice:**
mBART uses a sequence-to-sequence transformer architecture with language-specific tokens. Unlike traditional translation models that require separate models for each language pair, mBART uses a single model with language code prefixes (e.g., `ar_AR`, `en_XX`) to handle all language pairs.

---

## Evaluation Framework

The repository includes an evaluation framework for assessing OCR model performance.

### Evaluation Approach

The evaluation uses text-based metrics to assess model performance on invoice images with structured layouts and tables.

**Evaluation Methods:**

1. **Core OCR Metrics**: Standard OCR metrics (CER, WER, chrF, Exact Match) to measure content extraction accuracy
2. **Generation Metrics**: BLEU, ROUGE, METEOR to assess text generation quality
3. **Raw vs Cleaned Comparison**: Evaluates both raw and cleaned outputs for fair comparison
4. **Multiple Metrics**: Different metrics measure different aspects:
   - **Core OCR (Character-level)**: CER, chrF
   - **Core OCR (Word-level)**: WER, Exact Match
   - **Generation (Sequence-level)**: BLEU, ROUGE, METEOR

### Running Evaluation

**Step 1: Setup**

```bash
git clone <repository-url>
cd insurance_ai_engine
make setup-uv-and-sync
```

**Step 2: Configure Model URL**

```bash
export NANONETS_NANONETS_OCR2_3B_URL="http://your-gpu-server:8002/v1"
export OPENAI_API_KEY="DUMMY_API_KEY"
```

**Step 3: Run Evaluation**

```bash
# Run evaluation on all active models
make run-evaluate

# Or run directly
uv run python -m src.scripts.evaluate
```

The evaluation script:
- Loads the dataset from HuggingFace
- Runs OCR inference via API calls
- Computes metrics on raw and cleaned predictions
- Displays results in a table
- Saves JSON results

### Evaluation Metrics

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
| **CER** | Character | Core OCR | Percentage of characters that differ between prediction and ground truth | Lower is better (0.0 = perfect) | Primary OCR metric for character-level accuracy |
| **WER** | Word | Core OCR | Percentage of words that differ between prediction and ground truth | Lower is better (0.0 = perfect) | Primary OCR metric for word-level accuracy |
| **chrF** | Character | Core OCR | Harmonic mean of character precision and recall using n-grams | Higher is better (0-100 scale) | Character-level similarity with order awareness |
| **Exact Match** | Word | Core OCR | Percentage of samples where entire prediction exactly matches ground truth | Higher is better (0.0-1.0) | Assessing perfect accuracy rate |
| **BLEU** | Sequence | Generation | Measures n-gram precision between prediction and reference | Higher is better (0.0-1.0) | Text generation quality assessment |
| **ROUGE-1/2/L** | Sequence | Generation | Recall-oriented metrics (unigram, bigram, LCS) | Higher is better (0.0-1.0) | Text generation quality assessment |
| **METEOR** | Sequence | Generation | Harmonic mean with synonym matching and word order | Higher is better (0.0-1.0) | Text generation quality assessment |

### Evaluation Results

**Results Format:**

Each evaluation saves a JSON file to `results/` with pattern: `{model_name}_{timestamp}.json`

```json
{
  "model": "nanonets/Nanonets-OCR2-3B",
  "num_samples": 100,
  "timestamp": "2025-11-21T19:07:28.364252",
  "duration_seconds": 395.35,
  "duration_formatted": "6m 35s",
  "model_config": { ... },
  "dataset_info": { ... },
  "metrics": {
    "raw": {
      "cer": 0.87,
      "wer": 0.84,
      "chrf": { "score": 42.19 },
      "bleu": { "bleu": 0.19 },
      "rouge": { "rouge1": 0.56, "rouge2": 0.44, "rougeL": 0.52 },
      "meteor": { "meteor": 0.42 }
    },
    "cleaned": {
      "cer": 0.80,
      "wer": 1.03,
      "chrf": { "score": 45.84 },
      "bleu": { "bleu": 0.28 },
      "rouge": { "rouge1": 0.58, "rouge2": 0.51, "rougeL": 0.55 },
      "meteor": { "meteor": 0.41 }
    }
  },
  "samples": [ ... ]
}
```

**Visualization:**

![Evaluation Results Table](docs/images/evaluation_results_table.png)

![Evaluation Charts](docs/images/evaluation_charts.png)

### Dataset

The evaluation uses the `amaye15/invoices-google-ocr` dataset, which contains:
- **Images**: Invoice and document images (PNG format)
- **OCR Annotations**: Structured OCR data with bounding boxes and text
- **Labels**: Document type classification (Invoice, Receipt, Barcode, etc.)

### Text Cleaning

The evaluation pipeline includes two cleaning functions:

**`clean_html_markdown()`**: Removes HTML tags and markdown syntax
- Strips `<table>`, `<tr>`, `<td>` tags
- Removes markdown headers, lists, code blocks
- Preserves text content

**`clean_ocr_text()`**: OCR text normalization
- Unicode normalization (smart quotes, dashes, etc.)
- Whitespace normalization
- Punctuation spacing fixes
- Zero-width character removal

This cleaning ensures fair comparison between models that output structured formats and plain text.

### Translation Evaluation

The evaluation framework also supports translation model evaluation using standard machine translation metrics.

#### Translation Metrics

Translation evaluation uses metrics designed for machine translation quality assessment:

| Metric | Type | Description | Direction | Use Case |
|--------|------|-------------|-----------|----------|
| **BLEU** | Sequence | Measures n-gram precision (1-4 grams) between translation and reference. Includes brevity penalty for short translations | Higher is better (0.0-1.0) | Primary translation quality metric, widely used in MT evaluation |
| **METEOR** | Sequence | Harmonic mean of precision and recall with synonym matching. Considers word order and semantic similarity | Higher is better (0.0-1.0) | Better correlation with human judgment than BLEU, handles synonyms |
| **chrF** | Character | Character-level F-score measuring character n-gram overlap. Less affected by word segmentation differences | Higher is better (0-100 scale) | Useful for languages with different writing systems or word boundaries |
| **TER** | Sequence | Translation Error Rate - measures the minimum number of edits (insertions, deletions, substitutions, shifts) needed to transform translation into reference | Lower is better (0-100 scale, 0 = perfect) | Word-level error rate assessment, complementary to BLEU |

#### Translation Evaluation Results

Translation evaluation results are saved in JSON format similar to OCR evaluation. Example results from mBART model evaluation:

```json
{
  "model": "facebook/mbart-large-50-many-to-many-mmt",
  "model_type": "mbart",
  "num_samples": 100,
  "model_config": {
    "name": "facebook/mbart-large-50-many-to-many-mmt",
    "type": "mbart",
    "src_lang": "ar_AR",
    "target_lang": "en_XX",
    "max_workers": 3
  },
  "dataset_info": {
    "dataset_name": "Helsinki-NLP/opus-100",
    "split": "test",
    "num_samples": 100
  },
  "metrics": {
    "bleu": {
      "bleu": 0.1952
    },
    "meteor": {
      "meteor": 0.4703
    },
    "chrf": {
      "score": 43.5020
    },
    "ter": {
      "score": 73.3708
    }
  },
  "samples": [
    {
      "sample_index": 0,
      "source_text": "حسناً ، تبعاً للتقرير هناك ثلاث عبوات ماء مفقودة",
      "predicted_translation": "",
      "reference_translation": "Well, according to the report, there were three water bottles missing."
    }
  ]
}
```

**Visualization:**

![Translation Evaluation Results](docs/images/tr_eval.png)

#### Interpreting Translation Metrics

Translation metrics provide different perspectives on translation quality:

- **BLEU**: Focuses on precision - how many n-grams from the translation match the reference. Low scores may indicate poor quality or empty outputs.
- **METEOR**: Balances precision and recall with synonym awareness. Often correlates better with human judgment than BLEU.
- **chrF**: Character-level evaluation useful for morphologically rich languages or different writing systems.
- **TER**: Measures word-level errors - the minimum edits needed to match the reference. Lower scores indicate fewer errors.

#### Translation Dataset

Translation evaluation uses the `Helsinki-NLP/opus-100` dataset, which contains:
- Parallel text pairs in multiple languages
- High-quality human translations
- Various domains (news, legal, conversational, etc.)
- Language pairs: Arabic-English (ar-en), Czech-English (cs-en), and others

#### Running Translation Evaluation

**Note:** mBART models run locally in the FastAPI server process using the transformers library. They are loaded when the API server starts, not as a separate service.

```bash
# Set environment variables (only needed for OCR models)
export NANONETS_NANONETS_OCR2_3B_URL="http://localhost:8002"
export OPENAI_API_KEY="DUMMY_API_KEY"

# Start API server (mBART model loads automatically)
make start-api-server SERVER_PORT=8000

# Run translation evaluation (in another terminal)
make evaluate-translation
```

The translation evaluation:
- Loads parallel text pairs from the dataset
- Calls the translation API endpoint (mBART runs in the same server process)
- Computes BLEU, METEOR, chrF, and TER metrics
- Saves results to JSON files in `results/` directory

### OCR Evaluation Limitations and Future Enhancements

**Current Limitations:**
- Focuses on text-based metrics only
- Doesn't assess layout preservation, HTML/XML structure accuracy, or spatial relationships

**Future Enhancements:**
- Layout evaluation metrics
- HTML/XML parsing accuracy
- Spatial relationship metrics
- Table-specific evaluation

---

## System Architecture

The system follows a unified deployment architecture where all backend components run on a single L40 GPU instance on RunPod.io ($1/hour). Components can also be deployed separately for local development:

```
┌─────────────────────────────────────────────────────────────────┐
│              RunPod.io Cloud Platform (L40 GPU)                 │
│                    Cost: $1.00/hour                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Single L40 GPU Instance                                   │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  OCR Model (vLLM Server)                             │  │  │
│  │  │  - nanonets/Nanonets-OCR2-3B                         │  │  │
│  │  │  - OpenAI-compatible API                              │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  Translation Model (mBART)                          │  │  │
│  │  │  - facebook/mbart-large-50-many-to-many-mmt         │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  API Server (FastAPI)                               │  │  │
│  │  │  - OCR endpoints                                     │  │  │
│  │  │  - Translation endpoints                             │  │  │
│  │  │  - RunPod Proxy: *.proxy.runpod.net                 │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ HTTPS
                            │
┌───────────────────────────┼───────────────────────────────────┐
│                    Client Devices                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Web Interface (web/index.html)                          │ │
│  │  - Document OCR upload                                    │ │
│  │  - Text translation                                       │ │
│  │  - Results visualization                                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Evaluation Scripts (Local Machine)                      │ │
│  │  - Dataset loading                                        │ │
│  │  - Batch inference                                        │ │
│  │  - Metric computation                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

1. **Separation of Concerns**: Model serving, API server, and evaluation are separate components
2. **Cloud Deployment**: Backend deployed on RunPod.io for GPU access and cost-effectiveness
---

## Components

### API Server

The FastAPI server (`server/main.py`) provides a REST API for OCR and translation inference.

#### Endpoints

**Health & Information:**
- `GET /` - API information and active models
- `GET /health` - Health check
- `GET /models` - List all active OCR and translation models

**OCR Endpoints:**
- `POST /ocr/infer` - Single image/PDF inference (file upload)
- `POST /ocr/infer-base64` - Base64 encoded image inference
- `POST /ocr/batch` - Batch inference (multiple files)

**Translation Endpoints:**
- `POST /translation/translate` - Single text translation
- `POST /translation/batch` - Batch text translation

#### Running the API Server

```bash
# Start API server (default port 8003)
make start-api-server

# Or specify custom port
make start-api-server SERVER_PORT=8000
```

See `server/README.md` for API documentation.

### Web Interface

A Simple web interface (`web/`) for interacting with OCR and Translation APIs.

![Web Interface](docs/images/Web_image.png)

#### Features

- **Document OCR**: Upload images (PNG, JPEG, GIF, BMP, WebP) or PDF files to extract text
- **Text Translation**: Translate Text
- **Markdown Rendering**: Properly renders markdown and HTML content in OCR results (including tables)
- **Configurable API URL**: Set API endpoint via UI (saved in browser localStorage)

#### Using the Web Interface

**Option 1: Open directly in browser**


**Option 2: Serve with a local server**

```bash
# Using Python
cd web
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

**Configuration:**

The API server URL defaults to RunPod proxy endpoint but can be changed:
1. **Via UI**: Enter the URL in the "API Endpoint" field and click "Save"
2. **Via localStorage**: The URL is saved in browser localStorage for persistence

**For Local Development:**

If running the API server locally, update the API endpoint to:
- `http://localhost:8000` (default FastAPI port)
- Or your custom port if configured differently

---

## Deployment

### Deployment Architecture: RunPod.io

The backend API server and OCR models are deployed on [RunPod.io](https://www.runpod.io/), a cloud GPU platform.

#### Current Tesing Deployment Setup

**GPU Instance:**
- **GPU Type**: NVIDIA L40 GPU
- **Cost**: $1.00 per hour (pay-per-use)
- **Hosting**: Both the FastAPI backend server and OCR/translation models run on the same L40 GPU instance


### Local Deployment

For local development or testing:

#### Step 1: Clone and Setup

```bash
git clone <repository-url>
cd insurance_ai_engine
make setup-uv-and-sync
```

#### Step 2: Deploy OCR Model Server

```bash
# Deploy Nanonets OCR2 3B model (default port 8002)
make ocr-serve-nanonets [PORT=8002]

# Deploy Dots OCR model
make ocr-serve-dots [PORT=8002]

# Stop all vLLM servers
make stop
```

#### Step 3: Verify Deployment

```bash
# Check health
curl http://localhost:8002/health

# List models
curl http://localhost:8002/v1/models
```

#### Step 4: Start API Server

```bash
# Set environment variables
export NANONETS_NANONETS_OCR2_3B_URL="http://localhost:8002"
export OPENAI_API_KEY="DUMMY_API_KEY"

# Start API server (mBART translation model loads automatically in the server process)
make start-api-server SERVER_PORT=8000
```

### Production Considerations

- **GPU Memory**: Adjust `--gpu-memory-utilization` based on available VRAM
- **Concurrency**: Tune `--max-num-seqs` and `--max-num-batched-tokens` for your workload
- **Prefix Caching**: Enabled by default for better performance with repeated prompts
- **Process Management**: Consider using systemd or supervisor for production deployments
- **Load Balancing**: Use multiple API server instances behind a load balancer

---

## Configuration

### Model Configuration

Model configuration is managed through `src/config/models.yaml`. Each model entry includes:

**OCR Models:**
- `name`: Model identifier (e.g., `nanonets/Nanonets-OCR2-3B`)
- `prompt`: Instruction prompt for the OCR task
- `batch_size`: Number of samples processed per batch during evaluation
- `max_workers`: Maximum parallel API calls for batch inference
- `num_samples`: Number of samples to evaluate (optional)
- `active`: Whether the model is active (only active models are loaded)
- `save_results`: Whether to save evaluation results to JSON
- `results_dir`: Directory for saving results
- `url`: Optional explicit API URL (can also be set via environment variables)

**Translation Models:**
- `name`: Model identifier (e.g., `facebook/mbart-large-50-many-to-many-mmt`)
- `type`: Model type (`mbart` or `llm`)
- `model_path`: HuggingFace model path for mBART (e.g., `facebook/mbart-large-50-many-to-many-mmt`) or API URL for LLM
- `src_lang`: Default source language code (e.g., `ar_AR`, `en_XX`)
- `target_lang`: Default target language code (e.g., `en_XX`, `ar_AR`)
- `max_workers`: Maximum parallel workers for batch translation
- `active`: Whether the model is active

**Important:** mBART models (`type: mbart`) run locally in the FastAPI server process using the transformers library. They are loaded when the server starts - no separate service or URL configuration needed. LLM-based translation models (`type: llm`) require an API URL.

### Environment Variables

Model URLs can be configured via environment variables. The system automatically appends `/v1` for OpenAI-compatible APIs (like vLLM) if not present.

#### Required Environment Variables

**For OCR Models:**
- `{MODEL_NAME}_URL`: Model URL (e.g., `NANONETS_NANONETS_OCR2_3B_URL`)
- `MODEL_URL`: Fallback URL for all models (if model-specific URL not set)
- `OPENAI_API_KEY`: Required by OpenAI client (can be `DUMMY_API_KEY` for local vLLM)

**For Translation Models:**
- `{MODEL_NAME}_URL`: For LLM-based translation models (API endpoints)
- **Note:** mBART models run locally in the FastAPI server process using transformers - no URL needed

#### Setting Up Environment Variables

**Option 1: Using `.env` file (Recommended for Local Development)**

Create a `.env` file in the project root:

```bash
# OCR Model URLs (vLLM servers)
NANONETS_NANONETS_OCR2_3B_URL=http://localhost:8002
REDNOTE_HILAB_DOTS_OCR_URL=http://localhost:8002

# Fallback URL for all models
MODEL_URL=http://localhost:8002

# OpenAI API Key (required but can be dummy for local vLLM)
OPENAI_API_KEY=DUMMY_API_KEY

# Note: mBART translation models run locally in the FastAPI server process - no URL needed
```

**Option 2: Export in Shell**

```bash
export NANONETS_NANONETS_OCR2_3B_URL="http://localhost:8002"
export OPENAI_API_KEY="DUMMY_API_KEY"
```

**Option 3: Set in System Environment**

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export NANONETS_NANONETS_OCR2_3B_URL="http://localhost:8002"
export OPENAI_API_KEY="DUMMY_API_KEY"
```

**Important Notes:**
- The URL can be `http://localhost:8002` or `http://localhost:8002/` - the system will automatically add `/v1`
- You don't need to include `/v1` in the URL, but you can if you prefer
- `OPENAI_API_KEY` is required by the OpenAI client library but can be any dummy value for local vLLM servers
- For RunPod deployments, use the proxy URL: `https://your-pod-id.proxy.runpod.net`

---

## Usage Guide

### Quick Start: Complete Local Setup

1. **Start OCR Model Server** (Terminal 1):
   ```bash
   make ocr-serve-nanonets PORT=8002
   ```

2. **Set Environment Variables** (Terminal 2):
   ```bash
   export NANONETS_NANONETS_OCR2_3B_URL="http://localhost:8002"
   export OPENAI_API_KEY="DUMMY_API_KEY"
   # Note: mBART translation model runs locally in the API server - no URL needed
   ```

3. **Start API Server** (Terminal 2):
   ```bash
   make start-api-server SERVER_PORT=8000
   ```

4. **Open Web Interface**:
   ```bash
   cd web
   python -m http.server 8080
   ```
   Then open `http://localhost:8080` in your browser.

### Using OCR via API

**Single Image Inference:**

```bash
curl -X POST "http://localhost:8000/ocr/infer" \
  -F "file=@image.jpg" \
  -F "model_name=nanonets/Nanonets-OCR2-3B"
```

**PDF Document Inference:**

```bash
curl -X POST "http://localhost:8000/ocr/infer" \
  -F "file=@document.pdf" \
  -F "model_name=nanonets/Nanonets-OCR2-3B" \
  -F "pdf_dpi=200"
```

**Base64 Image Inference:**

```bash
curl -X POST "http://localhost:8000/ocr/infer-base64" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "base64_encoded_image_string",
    "model_name": "nanonets/Nanonets-OCR2-3B"
  }'
```

### Using Translation via API

**Single Text Translation:**

```bash
curl -X POST "http://localhost:8000/translation/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "src_lang": "en_XX",
    "target_lang": "ar_AR"
  }'
```

**Batch Translation:**

```bash
curl -X POST "http://localhost:8000/translation/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "World"],
    "src_lang": "en_XX",
    "target_lang": "ar_AR"
  }'
```

### Using the Web Interface

1. **Open the web interface** (see [Web Interface](#web-interface) section)
2. **Configure API Endpoint** (if not using default RunPod URL)
3. **For OCR**:
   - Click on "OCR" tab
   - Upload an image or PDF file
   - Click "Process OCR"
   - View results with rendered markdown/HTML
4. **For Translation**:
   - Click on "Translation" tab
   - Enter source text
   - Select source and target languages (required)
   - Click "Translate"
   - View translation results

---

## Evaluation Framework

The repository includes an evaluation framework for assessing OCR model performance.

### Evaluation Approach

The evaluation uses text-based metrics to assess model performance on insurance documents with structured layouts and tables.

**Evaluation Methods:**

1. **Core OCR Metrics**: Standard OCR metrics (CER, WER, chrF, Exact Match) to measure content extraction accuracy
2. **Generation Metrics**: BLEU, ROUGE, METEOR to assess text generation quality
3. **Raw vs Cleaned Comparison**: Evaluates both raw and cleaned outputs for fair comparison
4. **Multiple Metrics**: Different metrics measure different aspects:
   - **Core OCR (Character-level)**: CER, chrF
   - **Core OCR (Word-level)**: WER, Exact Match
   - **Generation (Sequence-level)**: BLEU, ROUGE, METEOR

### Running Evaluation

**Step 1: Setup**

```bash
git clone <repository-url>
cd insurance_ai_engine
make setup-uv-and-sync
```

**Step 2: Configure Model URL**

```bash
export NANONETS_NANONETS_OCR2_3B_URL="http://your-gpu-server:8002/v1"
export OPENAI_API_KEY="DUMMY_API_KEY"
```

**Step 3: Run Evaluation**

```bash
# Run evaluation on all active models
make run-evaluate

# Or run directly
uv run python -m src.scripts.evaluate
```

The evaluation script:
- Loads the dataset from HuggingFace
- Runs OCR inference via API calls
- Computes metrics on raw and cleaned predictions
- Displays results in a table
- Saves JSON results

### Evaluation Metrics

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
| **CER** | Character | Core OCR | Percentage of characters that differ between prediction and ground truth | Lower is better (0.0 = perfect) | Primary OCR metric for character-level accuracy |
| **WER** | Word | Core OCR | Percentage of words that differ between prediction and ground truth | Lower is better (0.0 = perfect) | Primary OCR metric for word-level accuracy |
| **chrF** | Character | Core OCR | Harmonic mean of character precision and recall using n-grams | Higher is better (0-100 scale) | Character-level similarity with order awareness |
| **Exact Match** | Word | Core OCR | Percentage of samples where entire prediction exactly matches ground truth | Higher is better (0.0-1.0) | Assessing perfect accuracy rate |
| **BLEU** | Sequence | Generation | Measures n-gram precision between prediction and reference | Higher is better (0.0-1.0) | Text generation quality assessment |
| **ROUGE-1/2/L** | Sequence | Generation | Recall-oriented metrics (unigram, bigram, LCS) | Higher is better (0.0-1.0) | Text generation quality assessment |
| **METEOR** | Sequence | Generation | Harmonic mean with synonym matching and word order | Higher is better (0.0-1.0) | Text generation quality assessment |

### Evaluation Results

**Results Format:**

Each evaluation saves a JSON file to `results/` with pattern: `{model_name}_{timestamp}.json`

```json
{
  "model": "nanonets/Nanonets-OCR2-3B",
  "num_samples": 100,
  "timestamp": "2025-11-21T19:07:28.364252",
  "duration_seconds": 395.35,
  "duration_formatted": "6m 35s",
  "model_config": { ... },
  "dataset_info": { ... },
  "metrics": {
    "raw": {
      "cer": 0.87,
      "wer": 0.84,
      "chrf": { "score": 42.19 },
      "bleu": { "bleu": 0.19 },
      "rouge": { "rouge1": 0.56, "rouge2": 0.44, "rougeL": 0.52 },
      "meteor": { "meteor": 0.42 }
    },
    "cleaned": {
      "cer": 0.80,
      "wer": 1.03,
      "chrf": { "score": 45.84 },
      "bleu": { "bleu": 0.28 },
      "rouge": { "rouge1": 0.58, "rouge2": 0.51, "rougeL": 0.55 },
      "meteor": { "meteor": 0.41 }
    }
  },
  "samples": [ ... ]
}
```

**Visualization:**

![Evaluation Results Table](docs/images/evaluation_results_table.png)

![Evaluation Charts](docs/images/evaluation_charts.png)

### Dataset

The evaluation uses the `amaye15/invoices-google-ocr` dataset, which contains:
- **Images**: Invoice and document images (PNG format)
- **OCR Annotations**: Structured OCR data with bounding boxes and text
- **Labels**: Document type classification (Invoice, Receipt, Barcode, etc.)

### Text Cleaning

The evaluation pipeline includes two cleaning functions:

**`clean_html_markdown()`**: Removes HTML tags and markdown syntax
- Strips `<table>`, `<tr>`, `<td>` tags
- Removes markdown headers, lists, code blocks
- Preserves text content

**`clean_ocr_text()`**: OCR text normalization
- Unicode normalization (smart quotes, dashes, etc.)
- Whitespace normalization
- Punctuation spacing fixes
- Zero-width character removal

---

## Development

### Project Structure

```
insurance_ai_engine/
├── src/
│   ├── config/           # Configuration loading (models.yaml)
│   ├── evaluation/       # Evaluation metrics and logic
│   ├── inference/        # OCR and translation inference via API
│   │   ├── ocr.py        # OCR inference logic
│   │   └── translation.py # Translation inference logic
│   ├── models/           # Model configuration models
│   ├── scripts/          # Evaluation scripts
│   │   ├── evaluate.py   # Main evaluation script
│   │   └── data_loader.py # Dataset loading utilities
│   └── utils.py          # Utility functions (image encoding, text cleaning)
├── server/               # FastAPI server for OCR and Translation API
│   ├── main.py          # API server implementation
│   └── README.md        # API documentation
├── web/                  # Web interface
│   ├── index.html       # Main HTML file
│   ├── styles.css       # Styling
│   └── app.js           # JavaScript logic
├── notebooks/           # Jupyter notebooks for experimentation
│   └── 01_experminets.ipynb # Evaluation experiments and visualizations
├── results/             # Evaluation results (JSON files)
├── data/                # Dataset cache directory
├── docs/                # Documentation and images
├── makefile             # Deployment and utility commands
└── pyproject.toml       # Python dependencies
```

### Adding a New OCR Model

1. **Add model configuration to `src/config/models.yaml`:**

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

2. **Add deployment command to `makefile`:**

```makefile
ocr-serve-your-model:
    uv run vllm serve your-model/name --host 0.0.0.0 --port $(PORT) ...
```

3. **Set environment variable for model URL:**

```bash
export YOUR_MODEL_NAME_URL="http://localhost:8002/v1"
```

### Adding a New Translation Model

1. **Add model configuration to `src/config/models.yaml`:**

```yaml
translation_models:
  - name: your-translation-model
    type: mbart  # or "llm"
    model_path: "path/to/model"  # or URL for LLM
    src_lang: "en_XX"
    target_lang: "ar_AR"
    max_workers: 4
    active: true
```

2. **Update translation inference code** if needed (for custom model types)

### Available Makefile Commands

View all available commands:

```bash
make help
```

**Common commands:**
- `make setup-uv-and-sync` - Setup uv and install dependencies
- `make ocr-serve-nanonets` - Deploy Nanonets OCR model
- `make ocr-serve-dots` - Deploy Dots OCR model
- `make start-api-server` - Start FastAPI server
- `make run-evaluate` - Run evaluation on active models
- `make stop` - Stop all vLLM servers
- `make clean` - Clean project artifacts

### Programmatic Usage

**OCR Evaluation:**

```python
from datasets import load_dataset
from src.config import load_config
from src.evaluation import evaluate_dataset

# Load configuration
config = load_config()

# Load dataset
dataset = load_dataset("amaye15/invoices-google-ocr", split="test")

# Run evaluation
results = evaluate_dataset(
    dataset=dataset,
    model_config=config.ocr_models[0],
    num_samples=100,
    dataset_name="amaye15/invoices-google-ocr",
    dataset_split="test"
)

# Access metrics
print(f"CER (cleaned): {results['metrics']['cleaned']['cer']}")
print(f"WER (cleaned): {results['metrics']['cleaned']['wer']}")
```

**OCR Inference:**

```python
from src.inference.ocr import infer
from src.config import load_config

config = load_config()
model_config = config.ocr_models[0]

# Encode image to base64
from src.utils import encode_image
from PIL import Image

image = Image.open("document.jpg")
img_base64 = encode_image(image)

# Run inference
result = infer(img_base64, model_config)
print(result)
```

**Translation:**

```python
from src.inference.translation import infer as translate
from src.config import load_config

config = load_config()
model_config = config.translation_models[0]

# Translate text
result = translate(
    "Hello, how are you?",
    model_config,
    src_lang="en_XX",
    target_lang="ar_AR"
)
print(result)
```

### Notebook Experiments

The `notebooks/01_experminets.ipynb` notebook contains:
- Dataset exploration and analysis
- Custom evaluation runs
- Visualization code for generating comparison charts
- Experimentation with different model configurations

Use this notebook as a reference for custom evaluations and visualizations.

---

## Troubleshooting

### Model Server Not Starting

**Symptoms:** vLLM server fails to start or crashes

**Solutions:**
- Check GPU availability: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Check port availability: `lsof -i :8002`
- Review vLLM logs for memory errors
- Reduce `--gpu-memory-utilization` if out of memory

### API Server Connection Issues

**Symptoms:** API server can't connect to OCR/translation models

**Solutions:**
- Verify model URLs are correct: `curl $MODEL_URL/health`
- Check environment variables are set: `env | grep URL`
- Ensure models are running and accessible
- Check firewall/network settings
- Verify API keys if using cloud services

### Evaluation Failing

**Symptoms:** Evaluation script errors or hangs

**Solutions:**
- Verify model URL is accessible: `curl $MODEL_URL/health`
- Check environment variables are set correctly
- Ensure dataset can be downloaded (HuggingFace access)
- Review API timeout settings in `src/inference/ocr.py`
- Check network connectivity to model servers

### High Memory Usage

**Symptoms:** System runs out of memory during evaluation

**Solutions:**
- Reduce `batch_size` in model configuration
- Lower `max_workers` for parallel API calls
- Use smaller `num_samples` for testing
- Process dataset in smaller chunks
- Close other applications using GPU memory

### Web Interface Not Loading

**Symptoms:** Web interface can't connect to API server

**Solutions:**
- Verify API server is running: `curl http://localhost:8000/health`
- Check API endpoint URL in web interface
- Verify CORS settings if serving from different origin
- Check browser console for errors
- Ensure API server allows requests from web interface origin

### Translation Errors

**Symptoms:** Translation returns errors or incorrect results

**Solutions:**
- Verify translation model is loaded and running
- Check language codes are correct (e.g., `ar_AR`, not `ar`)
- Ensure source and target languages are different
- Verify mBART model is loading correctly (check server startup logs)
- Ensure GPU/CUDA is available if running on GPU (mBART uses transformers with PyTorch)
- Check translation model configuration in `src/config/models.yaml`
