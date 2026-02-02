.PHONY: setup ingest serve eval chat clean test lint format help

help:
@echo "Mystic RAG - Available Commands"
@echo "================================"
@echo "make setup    - Install dependencies and pull models"
@echo "make ingest   - Process and embed documents"
@echo "make serve    - Start the FastAPI server"
@echo "make chat     - Interactive CLI chat"
@echo "make eval     - Run evaluation pipeline"
@echo "make test     - Run tests"
@echo "make clean    - Remove processed data and cache"

setup:
pip install -r requirements.txt
ollama pull llama3.2
ollama pull bge-base-en-v1.5
@echo "Setup complete!"

ingest:
python -m src.ingestion.main

serve:
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

chat:
python -m src.api.cli

eval:
python -m src.evaluation.main

test:
pytest tests/ -v

lint:
ruff check src/ tests/

format:
black src/ tests/
ruff check --fix src/ tests/

clean:
rm -rf data/processed/*
rm -rf qdrant_data/
rm -rf __pycache__
rm -rf .pytest_cache
find . -type d -name "__pycache__" -exec rm -rf {} +
@echo "Cleaned!"
