# 🔮 Mystic RAG

A production-ready RAG chatbot with Chinese Zodiac & Numerology knowledge.

## 🌐 Live Demo

- **Web UI**: https://rag.alphonzojonesjr.com
- **API Docs**: https://rag-api.alphonzojonesjr.com/docs

## ✨ Features

| Feature | Description |
|---------|-------------|
| Hybrid Retrieval | Vector search + BM25 |
| Cross-Encoder Reranking | 10x better precision |
| Dual LLM Support | Ollama (free) + Claude (quality) |
| Conversation Memory | Multi-turn chat |
| GPU Accelerated | Tesla P4 |

## 🚀 Quick Start (Local)
```bash
docker start qdrant
ollama serve
python -m src.api.cli
```

## 📁 GitHub

https://github.com/Newbigfonsz/rag-demo
