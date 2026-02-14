# 🎯 Mystic RAG - Interview Cheat Sheet

## 🔗 Quick Links
- **Live Demo**: https://rag.alphonzojonesjr.com
- **API Docs**: https://rag-api.alphonzojonesjr.com/docs
- **GitHub**: https://github.com/Newbigfonsz/rag-demo

---

## 🗣️ 30-Second Pitch

> "I built a production RAG system from scratch that combines semantic search with LLM generation. It features two-stage retrieval with cross-encoder reranking, supports both local and cloud LLMs, includes an analytics dashboard for monitoring costs and latency, and uses RAGAS evaluation to measure answer quality. It is deployed on my homelab with GPU acceleration and exposed via Cloudflare Tunnel with HTTPS."

---

## 🏗️ Architecture Flow
```
User Query → Embedding (Ollama) → Vector Search (Qdrant, top 15)
          → Cross-Encoder Reranking (top 5) → LLM Generation (Ollama/Claude)
          → Analytics Logging → Response with Sources
```

---

## 💡 Key Technical Decisions

### Why Two-Stage Retrieval?
> "Bi-encoders are fast but approximate. Cross-encoders are slow but precise. By retrieving 15 candidates quickly, then reranking to top 5, I get the best of both. Scores improved from 0.7 to 7.2 - a 10x improvement."

### Why Dual LLM Support?
> "Local LLMs are great for development. Cloud LLMs offer better quality. Supporting both shows I understand tradeoffs. Users can switch at runtime."

### Why Track Analytics?
> "In production, you need observability. I track latency, token usage, costs, and user feedback. This drives optimization decisions."

### Why RAGAS Evaluation?
> "RAG systems are hard to evaluate. RAGAS gives me faithfulness, relevancy, and context precision. My current score is 0.42 - context precision (0.19) needs work."

---

## 📊 Key Metrics

| Metric | Value | Talking Point |
|--------|-------|---------------|
| Documents | 27 | Custom knowledge base |
| Chunks | 402 | Markdown-aware chunking |
| Reranking Improvement | 0.7 → 7.2 | 10x better scores |
| Cost per Query | ~$0.003 | Claude Sonnet |
| RAGAS Score | 0.42 | Room to improve retrieval |

---

## 🎬 Demo Script (5 min)

1. **Open Web UI** - Show https://rag.alphonzojonesjr.com
2. **Ask Question** - "What does it mean to be a Dragon?"
3. **Follow-up** - "What careers suit that sign?" (shows memory)
4. **Feedback** - Click thumbs up to show human-in-the-loop
5. **API Docs** - Show Swagger at /docs
6. **GitHub** - Show clean code structure

---

## ❓ Likely Questions & Answers

### "Walk me through what happens when a user asks a question."
> "Query gets embedded → Search Qdrant for top 15 → Rerank to top 5 → Pass context to LLM → Generate answer → Log to analytics"

### "How do you handle hallucinations?"
> "System prompt constrains to context only. Reranking ensures relevant context. RAGAS faithfulness score measures groundedness."

### "How would you improve retrieval?"
> "Context precision is 0.19 - needs work. I would try: adjust chunk size, tune hybrid search weights, add query expansion, or fine-tune embeddings."

### "Why not use LangChain?"
> "I wanted to understand the fundamentals. LangChain abstracts away important details. Building from scratch taught me how embeddings, vector search, and reranking actually work."

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Embeddings | nomic-embed-text (Ollama) |
| Vector DB | Qdrant (self-hosted) |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.2 + Claude Sonnet |
| API | FastAPI |
| Web UI | Streamlit |
| Analytics | SQLite + Custom Dashboard |
| Infra | Docker, Tesla P4, Cloudflare |

---

## 🎯 Strengths to Emphasize

1. **End-to-end ownership** - Built everything from ingestion to deployment
2. **Production mindset** - Analytics, monitoring, Docker, HTTPS
3. **ML engineering** - Actual retrieval optimization, not just prompting
4. **Cost awareness** - Track and optimize API spend
5. **Evaluation rigor** - RAGAS metrics, not just vibes

---

**Good luck! You have got this! 🚀**
