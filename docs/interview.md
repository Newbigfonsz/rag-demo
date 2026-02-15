# 🎯 Mystic RAG - Interview Cheat Sheet

## 🔗 Quick Links
- **Live Demo**: https://rag.alphonzojonesjr.com
- **API Docs**: https://rag-api.alphonzojonesjr.com/docs
- **GitHub**: https://github.com/Newbigfonsz/rag-demo

---

## 🗣️ 30-Second Pitch

> "I built a production RAG system from scratch that combines semantic search with LLM generation. It features two-stage retrieval with cross-encoder reranking, agentic query handling with automatic clarification, query rewriting for follow-up questions, and automatic LLM fallback for resilience. It includes an analytics dashboard for monitoring costs and latency, uses RAGAS evaluation for quality metrics, and is deployed on my homelab with GPU acceleration via Cloudflare Tunnel."

---

## 🏗️ Architecture Flow
```
User Query 
    → Query Rewriting (expands follow-ups using context)
    → Agentic Analysis (clarify? search? multi-search?)
    → Embedding (Ollama nomic-embed-text)
    → Vector Search (Qdrant, top 15)
    → Cross-Encoder Reranking (top 5, 10x improvement)
    → LLM Generation (Ollama or Claude with auto-fallback)
    → Analytics Logging (latency, tokens, cost)
    → Response with Sources + Feedback Buttons
```

---

## 💡 Key Technical Decisions

### Why Two-Stage Retrieval?
> "Bi-encoders are fast but approximate. Cross-encoders are slow but precise. By retrieving 15 candidates quickly, then reranking to top 5, I get the best of both. Scores improved from 0.7 to 7.2 - a 10x improvement."

### Why Query Rewriting?
> "Follow-up questions like 'what careers suit that sign?' are vague without context. The system automatically rewrites them using conversation history, so 'that sign' becomes 'Dragon' based on the previous exchange."

### Why Agentic RAG?
> "Not all queries are equal. The agent analyzes each query and decides: ask for clarification if too vague, execute multi-search for comparisons, or do a standard search. This improves answer quality significantly."

### Why Auto-Fallback?
> "Production systems need resilience. If Ollama fails, the system automatically retries 3 times with exponential backoff, then falls back to Claude API. Users never see an error."

### Why Dual LLM Support?
> "Local LLMs are great for development and cost-sensitive use. Cloud LLMs offer better quality. Supporting both shows I understand tradeoffs. The system can switch at runtime."

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
| Vector Dimensions | 768 | nomic-embed-text |
| Reranking Improvement | 0.7 → 7.2 | 10x better scores |
| Avg Latency | ~4-16s | Depends on GPU/CPU |
| Cost per Query | ~$0.003 | Claude Sonnet |
| RAGAS Score | 0.42 | Room to improve retrieval |

---

## 🎬 Demo Script (5 min)

1. **Open Web UI** - https://rag.alphonzojonesjr.com
2. **Ask**: "What does it mean to be a Dragon?" → Shows RAG working
3. **Follow-up**: "What careers suit that sign?" → Query rewriting in action
4. **Vague query**: "hi" → Agent asks for clarification
5. **Comparison**: "Compare Rat and Ox" → Multi-search
6. **Click 👍** → Human feedback captured
7. **Show sidebar** → Analytics: queries, cost, feedback
8. **API Docs** - https://rag-api.alphonzojonesjr.com/docs
9. **Health endpoint** - /health shows component status

---

## ❓ Likely Questions & Answers

### "Walk me through what happens when a user asks a question."
> "First, if it's a follow-up, query rewriting expands it using conversation context. Then the agent analyzes it - is it too vague? Does it need multiple searches? For a standard query, we embed it with nomic-embed-text, search Qdrant for top 15 candidates, rerank with a cross-encoder to get top 5, pass those as context to the LLM, log everything to analytics, and return the answer with sources."

### "How do you handle failures?"
> "Three layers: retry logic with exponential backoff (3 attempts), automatic fallback from Ollama to Claude if local LLM fails, and graceful error handling that never crashes the user experience. There's also a /health endpoint that monitors all components."

### "How does query rewriting work?"
> "It's rule-based for speed. The system detects pronouns and follow-up phrases like 'that sign' or 'tell me more', then looks at recent conversation history to find the topic (e.g., 'Dragon') and rewrites the query. So 'what careers suit that sign?' becomes 'what careers suit Dragon?'"

### "What's agentic about your RAG?"
> "The agent makes decisions about how to answer. If someone just says 'hi', it asks for clarification rather than searching. For comparisons like 'Rat vs Ox', it executes multiple searches and synthesizes. This improves answer quality for edge cases."

### "How do you handle hallucinations?"
> "System prompt constrains answers to provided context only. Reranking ensures relevant context. RAGAS faithfulness score measures groundedness. The score of 0.41 tells me there's room to improve."

### "How would you improve retrieval?"
> "Context precision is 0.19 - that's the bottleneck. I'd try: adjusting chunk size and overlap, tuning hybrid search weights between vector and BM25, adding query expansion, or fine-tuning the embedding model on domain-specific data."

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Embeddings | nomic-embed-text (Ollama) |
| Vector DB | Qdrant (self-hosted) |
| Reranker | ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.2 (local) + Claude Sonnet (cloud) |
| API | FastAPI |
| Web UI | Streamlit |
| Analytics | SQLite + custom dashboard |
| Evaluation | RAGAS metrics |
| Infra | Docker, Tesla P4 GPU, Cloudflare Tunnel |

---

## 🆕 Advanced Features

| Feature | Description |
|---------|-------------|
| **Query Rewriting** | Expands follow-up questions using conversation context |
| **Agentic RAG** | Decides: clarify, search, or multi-search based on query |
| **Auto-Fallback** | Ollama → Claude with 3x retry + exponential backoff |
| **Health Endpoint** | /health monitors Qdrant, Ollama, Claude availability |
| **Feedback Loop** | 👍👎 buttons capture user feedback for improvement |

---

## 🎯 Strengths to Emphasize

1. **End-to-end ownership** - Built everything from ingestion to deployment
2. **Production mindset** - Analytics, monitoring, health checks, fallback
3. **ML engineering** - Actual retrieval optimization, not just prompting
4. **Resilience** - Retry logic, auto-fallback, graceful degradation
5. **Evaluation rigor** - RAGAS metrics, not just vibes
6. **Cost awareness** - Track and optimize API spend

---

## 🚀 Commands Reference
```bash
# Local testing (Windows)
docker start qdrant
ollama serve
python -m src.api.cli
streamlit run src/ui/app.py

# RAGAS Evaluation
python -m src.evaluation.ragas_eval

# Production (Linux server)
cd ~/rag-demo
git pull
sudo docker compose down
sudo docker compose up -d --build
sudo docker compose exec api python -m src.ingestion.main
```

---

**Good luck! You built something real. Go get that job! 🚀**
