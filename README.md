# 🔮 Mystic RAG

A production-ready RAG (Retrieval-Augmented Generation) chatbot demonstrating modern ML/AI engineering practices.

**Domain**: Chinese Zodiac & Numerology knowledge base

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Vector search + BM25 keyword matching |
| **Cross-Encoder Reranking** | Two-stage retrieval for higher precision |
| **Conversation Memory** | Multi-turn chat with follow-up questions |
| **Multiple Interfaces** | CLI, REST API, Web UI |
| **Evaluation Pipeline** | Automated quality metrics |
| **Docker Deployment** | One-command deployment |

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interfaces                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   CLI Chat      │   FastAPI       │   Streamlit Web UI          │
│   (Terminal)    │   (REST API)    │   (Browser)                 │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         └─────────────────┼───────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Retrieval  │───▶│   Reranking  │───▶│  Generation  │      │
│  │  (Vector +   │    │   (Cross-    │    │   (Llama 3.2 │      │
│  │   Qdrant)    │    │   Encoder)   │    │   + Memory)  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│     Qdrant      │ │   Ollama    │ │  Knowledge Base │
│  (Vector DB)    │ │ (Local LLM) │ │  (27 Markdown   │
│   402 vectors   │ │  bge-m3 +   │ │   documents)    │
│                 │ │  llama3.2   │ │                 │
└─────────────────┘ └─────────────┘ └─────────────────┘
```

## 🚀 Quick Start
```bash
# Clone
git clone https://github.com/Newbigfonsz/rag-demo.git
cd rag-demo

# Setup
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate
pip install -r requirements.txt

# Start dependencies
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
ollama pull llama3.2
ollama pull nomic-embed-text

# Ingest documents
python -m src.ingestion.main

# Run chat
python -m src.api.cli
```

## 💬 Demo
```
You: What does it mean to be a Dragon?
Assistant: Dragons are known for their confidence, charisma, and leadership...

You: What careers suit that sign?  ← Follow-up using memory!
Assistant: Based on our discussion about Dragons, careers include...

You: memory  ← View conversation history
Memory: 2 exchanges
  You: What does it mean to be a Dragon?...
  Bot: Dragons are known for their confidence...
```

## 🔧 Interfaces

| Interface | Command | URL |
|-----------|---------|-----|
| **CLI** | `python -m src.api.cli` | Terminal |
| **API** | `uvicorn src.api.main:app --reload` | http://localhost:8000/docs |
| **Web UI** | `streamlit run src/ui/app.py` | http://localhost:8501 |
| **Evaluation** | `python -m src.evaluation.main` | Terminal |

## 📊 Performance

| Metric | Value |
|--------|-------|
| Documents | 27 |
| Chunks | 402 |
| Avg Latency | ~1.3s |
| Embedding Model | nomic-embed-text (768d) |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.2 (3B) |

## 📁 Project Structure
```
rag-demo/
├── data/
│   ├── raw/                 # 27 source documents
│   └── processed/           # Chunked data
├── src/
│   ├── ingestion/           # Document processing pipeline
│   ├── retrieval/           # Vector search + reranking
│   ├── generation/          # LLM with memory
│   ├── evaluation/          # Quality metrics
│   ├── api/                 # FastAPI + CLI
│   └── ui/                  # Streamlit
├── docs/
│   └── decisions.md         # Architecture Decision Records
├── docker-compose.yml       # One-command deployment
└── requirements.txt
```

## 🎯 Key Design Decisions

See [docs/decisions.md](docs/decisions.md) for detailed ADRs.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | Qdrant | Hybrid search support, no cloud dependency |
| Embeddings | nomic-embed-text | Open source, runs locally |
| LLM | Llama 3.2 | Free, local, good quality |
| Reranking | Cross-encoder | 10x better relevance scores |
| Chunking | Markdown-aware | Preserves document structure |

## 🛠️ Tech Stack

- **Python 3.11+**
- **Qdrant** - Vector database
- **Ollama** - Local LLM runtime
- **FastAPI** - REST API
- **Streamlit** - Web UI
- **Sentence Transformers** - Reranking

## 📄 License

MIT
