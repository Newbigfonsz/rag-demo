# Architecture Decision Records

This document tracks key architectural decisions, alternatives considered, and rationale.

---

## ADR-001: Embedding Model Selection

**Date**: 2024-XX-XX  
**Status**: Decided

### Context
Need an embedding model for semantic search. Must balance quality, speed, and ability to run locally.

### Options Considered

| Model | Dimensions | Quality (MTEB) | Local? | Cost |
|-------|------------|----------------|--------|------|
| OpenAI text-embedding-3-small | 1536 | High | No | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | Higher | No | $0.13/1M tokens |
| BGE-base-en-v1.5 | 768 | Good | Yes | Free |
| BGE-large-en-v1.5 | 1024 | Better | Yes | Free |
| E5-base | 768 | Good | Yes | Free |

### Decision
**BGE-base-en-v1.5** via Ollama

### Rationale
- Runs entirely locally (no API costs, no rate limits)
- Good quality for domain-specific content
- Fast inference on CPU
- Demonstrates understanding of open-source alternatives
- For a demo, local = reliable (no API failures during presentation)

### Consequences
- Slightly lower quality than OpenAI embeddings
- Need to manage Ollama setup
- Positive: Shows ML engineering depth by choosing and justifying open-source

---

## ADR-002: Vector Database Selection

**Date**: 2024-XX-XX  
**Status**: Decided

### Context
Need a vector database for storing and searching embeddings.

### Options Considered

| Database | Hybrid Search | Ease of Setup | Scalability |
|----------|--------------|---------------|-------------|
| Qdrant | Yes | Docker one-liner | High |
| ChromaDB | No | pip install | Medium |
| Pinecone | Yes | Cloud only | High |
| pgvector | Manual | Postgres needed | High |
| FAISS | No | In-memory | Low |

### Decision
**Qdrant** (local Docker)

### Rationale
- Native hybrid search support (vector + keyword)
- Simple Docker setup
- REST API for debugging
- No cloud dependency for demo
- Good documentation

### Consequences
- Need Docker installed
- More setup than ChromaDB
- Positive: Production-realistic choice

---

## ADR-003: Chunking Strategy

**Date**: 2024-XX-XX  
**Status**: Decided

### Context
Documents need to be split into chunks for embedding. Strategy affects retrieval quality.

### Options Considered

| Strategy | Pros | Cons |
|----------|------|------|
| Fixed size (512 tokens) | Simple, predictable | May split mid-concept |
| Recursive with overlap | Preserves context | More chunks |
| Semantic chunking | Best boundaries | Complex, slow |
| Markdown headers | Natural structure | Depends on doc format |

### Decision
**Markdown header-aware + recursive fallback**, 512 tokens with 50 token overlap

### Rationale
- Our corpus is markdown with clear headers
- Headers provide natural semantic boundaries
- Recursive fallback handles sections that are too long
- Overlap prevents losing context at boundaries

### Consequences
- Need custom splitter logic
- More sophisticated than basic approaches
- Positive: Shows understanding of chunking impact on retrieval

---

## ADR-004: Retrieval Strategy

**Date**: 2024-XX-XX  
**Status**: Decided

### Context
Pure vector search may miss keyword matches; pure keyword search misses semantics.

### Decision
**Hybrid search (vector + BM25) with cross-encoder reranking**

### Rationale
1. Hybrid captures both semantic similarity and keyword matches
2. Cross-encoder reranking significantly improves precision
3. Two-stage retrieval (retrieve many, rerank few) is production best practice

### Implementation
1. Retrieve top-20 with hybrid search (alpha=0.7 vector, 0.3 BM25)
2. Rerank with cross-encoder to top-5
3. Return top-5 for generation

### Consequences
- Increased latency from reranking
- More complexity
- Positive: Demonstrates understanding of retrieval optimization

---

## ADR-005: LLM Selection

**Date**: 2024-XX-XX  
**Status**: Decided

### Context
Need an LLM for answer generation. Must run locally for demo reliability.

### Options Considered

| Model | Quality | Speed | Memory |
|-------|---------|-------|--------|
| GPT-4 | Excellent | Fast | N/A (API) |
| Llama 3.2 8B | Good | Medium | 8GB |
| Mistral 7B | Good | Medium | 8GB |
| Phi-3 | Decent | Fast | 4GB |

### Decision
**Llama 3.2** via Ollama

### Rationale
- Latest Llama with good instruction following
- Runs locally via Ollama
- Good balance of quality and speed
- No API dependency during demo

### Consequences
- Need GPU or fast CPU for reasonable speed
- Quality below GPT-4
- Positive: Demonstrates full local stack

---

## ADR-006: Evaluation Framework

**Date**: 2024-XX-XX  
**Status**: Decided

### Decision
**RAGAS** for generation metrics + custom retrieval metrics

### Metrics Tracked
1. **Retrieval**
   - Precision@k
   - MRR (Mean Reciprocal Rank)
   - Recall@k

2. **Generation** (RAGAS)
   - Faithfulness
   - Answer Relevance
   - Context Precision

3. **Operational**
   - Latency (p50, p95)
   - Token usage

### Rationale
- RAGAS is the standard for RAG evaluation
- Custom retrieval metrics give more granular insight
- Operational metrics matter for production
