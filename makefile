.PHONY: run-nanonets stop help

PORT ?= 8001

run-nanonets:
	uv run vllm serve nanonets/Nanonets-OCR2-3B --host 0.0.0.0 --port $(PORT) --max-num-batched-tokens 131072 --max-model-len 100000

stop:
	@pkill -f "vllm serve" || echo "No vLLM process found"

help:
	@echo "Available models:"
	@echo "  make run-nanonets [PORT=<port>]  - Run Nanonets OCR2 3B model (OCR for document processing)"
	@echo ""
	@echo "Other commands:"
	@echo "  make stop                        - Stop vLLM serve"
	@echo "  make help                        - Show this help"

