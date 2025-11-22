.PHONY: help start-api-server ocr-serve-nanonets ocr-serve-dots stop \
        evaluate-ocr evaluate-translation setup-uv-and-sync clean
SHELL := /bin/bash
PORT ?= 8002                    
SERVER_PORT ?= 8003            
OCR_DATASET ?= amaye15/invoices-google-ocr
OCR_SPLIT ?= test
OCR_NUM_SAMPLES ?= 100
TRANSLATION_DATASET ?= Helsinki-NLP/opus-100
TRANSLATION_CONFIG ?= ar-en     
TRANSLATION_SPLIT ?= test
TRANSLATION_NUM_SAMPLES ?= 100


start-api-server:
	@echo "Starting FastAPI server on port $(SERVER_PORT)..."
	@export PATH="$$HOME/.local/bin:$$PATH" && \
	uv run uvicorn server.main:app --reload --host 0.0.0.0 --port $(SERVER_PORT)
	@echo "FastAPI server started!"

ocr-serve-nanonets:
	@echo "Starting Nanonets OCR2-3B model server on port $(PORT)..."
	@export PATH="$$HOME/.local/bin:$$PATH" && \
	VLLM_LOGGING_LEVEL=INFO uv run vllm serve nanonets/Nanonets-OCR2-3B \
		--host 0.0.0.0 \
		--port $(PORT) \
		--gpu-memory-utilization 0.85 \
		--async-scheduling \
		--max-num-seqs 32 \
		--max-num-batched-tokens 16384 \
		--enable-prefix-caching \
		--generation-config vllm \
		--block-size 16

ocr-serve-dots:
	@echo "Starting Dots OCR model server on port $(PORT)..."
	@export PATH="$$HOME/.local/bin:$$PATH" && \
	CUDA_LAUNCH_BLOCKING=1 \
	TORCH_USE_CUDA_DSA=1 \
	VLLM_LOGGING_LEVEL=INFO uv run vllm serve rednote-hilab/dots.ocr \
		--host 0.0.0.0 \
		--port $(PORT) \
		--trust-remote-code \
		--async-scheduling \
		--gpu-memory-utilization 0.75 \
		--max-num-seqs 32 \
		--enable-prefix-caching \
		--enforce-eager \
		--generation-config vllm

stop:
	@echo "Stopping vLLM model servers..."
	@PIDS=$$(ps aux | grep "[v]llm serve" | awk '{print $$2}' | tr '\n' ' '); \
	if [ -z "$$PIDS" ]; then \
		echo "No vLLM processes found"; \
	else \
		echo "Found vLLM processes: $$PIDS"; \
		kill $$PIDS 2>/dev/null || true; \
		sleep 2; \
		REMAINING=$$(ps aux | grep "[v]llm serve" | awk '{print $$2}' | tr '\n' ' '); \
		if [ -n "$$REMAINING" ]; then \
			echo "Force killing remaining processes: $$REMAINING"; \
			kill -9 $$REMAINING 2>/dev/null || true; \
		fi; \
		echo "vLLM processes stopped"; \

evaluate-ocr:
	@echo "Running OCR evaluation..."
	@echo "  Dataset: $(OCR_DATASET)"
	@echo "  Split: $(OCR_SPLIT)"
	@echo "  Samples: $(OCR_NUM_SAMPLES)"
	@uv run python -m src.scripts.evaluate \
		--task-type ocr \
		--dataset $(OCR_DATASET) \
		--split $(OCR_SPLIT) \
		--num-samples $(OCR_NUM_SAMPLES)

evaluate-translation:
	@echo "Running translation evaluation..."
	@echo "  Dataset: $(TRANSLATION_DATASET)"
	@echo "  Config: $(TRANSLATION_CONFIG)"
	@echo "  Split: $(TRANSLATION_SPLIT)"
	@echo "  Samples: $(TRANSLATION_NUM_SAMPLES)"
	@uv run python -m src.scripts.evaluate \
		--task-type translation \
		--dataset $(TRANSLATION_DATASET) \
		--dataset-config $(TRANSLATION_CONFIG) \
		--split $(TRANSLATION_SPLIT) \
		--num-samples $(TRANSLATION_NUM_SAMPLES)


setup-uv-and-sync:
	@echo "Setting up uv package manager..."
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
	fi
	@echo "Syncing project dependencies..."
	@export PATH="$$HOME/.local/bin:$$PATH" && uv sync
	@echo "Setup complete!"

clean:
	@echo "Cleaning project (removing .venv, .cache, __pycache__, etc.)..."
	@rm -rf .venv
	@rm -rf .cache
	@rm -rf .local
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@echo "Project cleaned!"

help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║         Insurance AI Engine - Available Commands               ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo " SERVER MANAGEMENT"
	@echo "  ──────────────────────────────────────────────────────────────"
	@echo "  make start-api-server [SERVER_PORT=<port>]"
	@echo "    Start FastAPI server (default port: $(SERVER_PORT))"
	@echo ""
	@echo "  make ocr-serve-nanonets [PORT=<port>]"
	@echo "    Start Nanonets OCR2-3B model server (default port: $(PORT))"
	@echo ""
	@echo "  make ocr-serve-dots [PORT=<port>]"
	@echo "    Start Dots OCR model server (default port: $(PORT))"
	@echo ""
	@echo "  make stop"
	@echo "    Stop all running vLLM model servers"
	@echo ""
	@echo "  MODEL EVALUATION"
	@echo "  ──────────────────────────────────────────────────────────────"
	@echo "  make evaluate-ocr [OCR_DATASET=<dataset>] [OCR_SPLIT=<split>] [OCR_NUM_SAMPLES=<n>]"
	@echo "    Evaluate OCR models"
	@echo "    Defaults: dataset=$(OCR_DATASET), split=$(OCR_SPLIT), samples=$(OCR_NUM_SAMPLES)"
	@echo ""
	@echo "  make evaluate-translation [TRANSLATION_DATASET=<dataset>] [TRANSLATION_CONFIG=<lang-pair>]"
	@echo "                          [TRANSLATION_SPLIT=<split>] [TRANSLATION_NUM_SAMPLES=<n>]"
	@echo "    Evaluate translation models"
	@echo "    Defaults: dataset=$(TRANSLATION_DATASET), config=$(TRANSLATION_CONFIG)"
	@echo "             split=$(TRANSLATION_SPLIT), samples=$(TRANSLATION_NUM_SAMPLES)"
	@echo ""
	@echo "  Examples:"
	@echo "    make evaluate-ocr OCR_NUM_SAMPLES=500"
	@echo "    make evaluate-ocr OCR_DATASET=custom/dataset OCR_NUM_SAMPLES=200"
	@echo "    make evaluate-translation TRANSLATION_CONFIG=de-en TRANSLATION_NUM_SAMPLES=1000"
	@echo ""
	@echo " DEVELOPMENT"
	@echo "  ──────────────────────────────────────────────────────────────"
	@echo "  make setup-uv-and-sync"
	@echo "    Install uv package manager and sync project dependencies"
	@echo ""
	@echo "  make clean"
	@echo "    Remove all build artifacts, cache files, and virtual environment"
	@echo ""
	@echo "  make help"
	@echo "    Show this help message"
	@echo ""
