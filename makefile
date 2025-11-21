.PHONY: ocr-serve-nanonets stop help clean setup-uv-and-sync ocr-serve-dots

SHELL := /bin/bash
PORT ?= 8002
SERVER_PORT ?= 8081

setup-uv-and-sync:
	curl -LsSf https://astral.sh/uv/install.sh | sh && \
	export PATH="$$HOME/.local/bin:$$PATH" && \
	uv --version && \
	uv sync

ocr-serve-nanonets:
	export PATH="$$HOME/.local/bin:$$PATH" && \
	VLLM_LOGGING_LEVEL=INFO uv run vllm serve nanonets/Nanonets-OCR2-3B --host 0.0.0.0 --port $(PORT) --max-num-batched-tokens 131072 --max-model-len 100000 --gpu-memory-utilization 0.75 --async-scheduling --max-num-seqs 32 --enable-prefix-caching

ocr-serve-dots:
	export PATH="$$HOME/.local/bin:$$PATH" && \
	VLLM_LOGGING_LEVEL=INFO uv run vllm serve rednote-hilab/dots.ocr --host 0.0.0.0 --port $(PORT) --trust-remote-code --async-scheduling --gpu-memory-utilization 0.75 --max-num-seqs 32 --enable-prefix-caching

stop:
	@pkill -f "vllm serve" || echo "No vLLM process found"
start-server:
	uvicorn server.main:app --host 0.0.0.0 --port $(SERVER_PORT)
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
	@echo "Other commands:"
	@echo "  make stop                        - Stop vLLM serve"
	@echo "  make clean                       - Clean entire project (venv, cache, build artifacts)"
	@echo "  make help                        - Show this help"
	@echo "  make setup-uv-and-sync           - helper utility to setup uv on ubuntu and sync the project"
