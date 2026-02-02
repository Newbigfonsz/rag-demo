# Mystic RAG: Numerology & Chinese Zodiac Intelligence

> A production-grade RAG system demonstrating retrieval optimization, answer grounding, and systematic evaluation — built on a unique domain of numerology and Chinese zodiac wisdom.

## What This Demonstrates

This project showcases ML/AI engineering skills through a complete RAG pipeline:

- **Hybrid Search**: Vector similarity + BM25 keyword matching
- **Reranking**: Cross-encoder model for improved relevance
- **Evaluation Pipeline**: Automated metrics with RAGAS
- **Citation Extraction**: Grounded answers with source attribution
- **Cost/Latency Tracking**: Production-ready observability

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Ingestion     │────▶│    Retrieval    │────▶│   Generation    │
│   Pipeline      │     │  (Hybrid+Rerank)│     │   + Citation    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Dashboard                          │
│  • Retrieval Precision@k  • Answer Faithfulness  • Latency/Cost │
└─────────────────────────────────────────────────────────────────┘
```

## Results

| Metric | Baseline | Current | Improvement |
|--------|----------|---------|-------------|
| Retrieval Precision@5 | - | - | - |
| MRR | - | - | - |
| Answer Faithfulness | - | - | - |
| Avg Latency (ms) | - | - | - |

*Updated as evaluation runs complete*

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- 8GB+ RAM recommended

### Setup

```bash
# Clone the repo
git clone https://github.com/Newbigfonsz/rag-demo.git
cd rag-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull required models
ollama pull llama3.2
ollama pull bge-base-en-v1.5

# Run setup
make setup
```

### Usage

```bash
# Ingest documents
make ingest

# Start the API
make serve

# Run evaluation
make eval

# Interactive chat
make chat
```

## Project Structure

```
rag-demo/
├── src/
│   ├── ingestion/      # Document processing & chunking
│   ├── retrieval/      # Hybrid search & reranking
│   ├── generation/     # LLM response & citation
│   ├── evaluation/     # RAGAS metrics & tracking
│   └── api/            # FastAPI endpoints
├── data/
│   ├── raw/            # Source documents
│   └── processed/      # Chunked & embedded data
├── docs/
│   └── decisions.md    # Architecture Decision Records
├── tests/
├── scripts/
└── notebooks/          # Exploration & analysis
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embeddings | BGE-base-en-v1.5 | Open source, good quality, runs locally |
| Vector DB | Qdrant | Simple setup, hybrid search support |
| LLM | Llama 3.2 (via Ollama) | Open source, runs locally |
| Reranker | Cross-encoder | Significant retrieval improvement |
| Evaluation | RAGAS | Standard RAG evaluation framework |
| API | FastAPI | Async, automatic docs |

## Example Queries

```
"What does it mean if I'm a Dragon with life path 7?"
"Are Rabbit and Snake compatible?"  
"What are the characteristics of master number 22?"
"I was born in 1988, what's my Chinese zodiac and what careers suit me?"
```

## License

MIT

---

*Built to demonstrate ML/AI engineering skills. Domain content generated for educational purposes.*
