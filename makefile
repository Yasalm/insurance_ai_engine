.PHONY: ocr-serve-nanonets stop help clean setup-uv-and-sync ocr-serve-dots evaluate evaluate-ocr evaluate-translation

SHELL := /bin/bash
PORT ?= 8002
SERVER_PORT ?= 8081
OCR_DATASET ?= amaye15/invoices-google-ocr
TRANSLATION_DATASET ?= Helsinki-NLP/opus-100
TRANSLATION_CONFIG ?= ar-en
NUM_SAMPLES ?= 100
SPLIT ?= test


start-api-server:
	@echo "Starting API server..."
	uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
	@echo "Server API started!"

ocr-serve-nanonets:
	export PATH="$$HOME/.local/bin:$$PATH" && \
	VLLM_LOGGING_LEVEL=INFO uv run vllm serve nanonets/Nanonets-OCR2-3B --host 0.0.0.0 --port $(PORT) --gpu-memory-utilization 0.95 --async-scheduling --max-num-seqs 32 --max-num-batched-tokens 16384 --enable-prefix-caching --generation-config vllm --block-size 16 

ocr-serve-dots:
	export PATH="$$HOME/.local/bin:$$PATH" && \
	CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 VLLM_LOGGING_LEVEL=INFO uv run vllm serve rednote-hilab/dots.ocr --host 0.0.0.0 --port $(PORT) --trust-remote-code --async-scheduling --gpu-memory-utilization 0.75 --max-num-seqs 32 --enable-prefix-caching --enforce-eager --generation-config vllm
stop:
	@echo "Stopping vLLM servers..."
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
	uv run python -m src.scripts.evaluate --task-type ocr --dataset $(OCR_DATASET) --split $(SPLIT) --num-samples $(NUM_SAMPLES)

evaluate-translation:
	uv run python -m src.scripts.evaluate --task-type translation --dataset $(TRANSLATION_DATASET) --dataset-config $(TRANSLATION_CONFIG) --split $(SPLIT) --num-samples $(NUM_SAMPLES)

setup-uv-and-sync:
	@echo "Setting up uv package manager..."
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.local/bin:$$PATH"; \
	fi
	@echo "Syncing project dependencies..."
	export PATH="$$HOME/.local/bin:$$PATH" && uv sync
	@echo "Setup complete!"

clean:
	@echo "Cleaning project (removing .venv, .cache, .local, __pycache__, .pyc files)..."
	rm -rf .venv
	rm -rf .cache
	rm -rf .local
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@echo "Project cleaned!"

help:
	@echo "Available models:"
	@echo "  make ocr-serve-nanonets [PORT=<port>]  - Run Nanonets OCR2 3B model (OCR for document processing)"
	@echo "  make ocr-serve-dots [PORT=<port>]  - Run Dots OCR model (OCR for document processing)"
	@echo ""
	@echo "Evaluation commands:"
	@echo "  make evaluate-ocr [OCR_DATASET=<dataset>] [NUM_SAMPLES=<n>] [SPLIT=<split>]"
	@echo "                    - Run OCR evaluation (default: $(OCR_DATASET), $(NUM_SAMPLES) samples)"
	@echo "  make evaluate-translation [TRANSLATION_DATASET=<dataset>] [TRANSLATION_CONFIG=<config>] [NUM_SAMPLES=<n>] [SPLIT=<split>]"
	@echo "                    - Run translation evaluation (default: $(TRANSLATION_DATASET), config: $(TRANSLATION_CONFIG), $(NUM_SAMPLES) samples)"
	@echo ""
	@echo "  Examples:"
	@echo "    make evaluate-ocr NUM_SAMPLES=500"
	@echo "    make evaluate-ocr OCR_DATASET=custom/dataset NUM_SAMPLES=200"
	@echo "    make evaluate-translation TRANSLATION_CONFIG=de-en NUM_SAMPLES=1000"
	@echo ""
	@echo "Other commands:"
	@echo "  make stop                        - Stop vLLM serve"
	@echo "  make setup-uv-and-sync          - Setup uv package manager and sync dependencies"
	@echo "  make clean                      - Clean entire project (venv, cache, build artifacts)"
	@echo "  make help                       - Show this help"
