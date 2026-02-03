# Architecture Decision Records

## ADR-001: Vector Database Selection
**Decision**: Qdrant  
**Rationale**: Supports hybrid search (vector + BM25), runs locally via Docker, no API costs  
**Alternatives Considered**: Pinecone (cloud-only), ChromaDB (limited hybrid support)

## ADR-002: Embedding Model
**Decision**: nomic-embed-text via Ollama  
**Rationale**: Open source, runs locally, 768 dimensions, good quality  
**Alternatives Considered**: OpenAI (costs money), BGE-M3 (NaN issues on some texts)

## ADR-003: LLM Selection
**Decision**: Llama 3.2 (3B) via Ollama  
**Rationale**: Free, runs locally, good instruction following  
**Alternatives Considered**: GPT-4 (costs), Mistral (similar quality)

## ADR-004: Two-Stage Retrieval
**Decision**: Vector search → Cross-encoder reranking  
**Rationale**: 
- Initial retrieval: Fast but approximate (top 15)
- Reranking: Slow but precise (top 5)
- Result: 10x improvement in relevance scores

**Evidence**:
```
Before (vector): Score 0.73
After (rerank):  Score 7.25
```

## ADR-005: Conversation Memory
**Decision**: In-memory message history with sliding window  
**Rationale**: 
- Enables follow-up questions ("What careers suit that sign?")
- Last 10 exchanges kept for context
- Simple, no external storage needed

## ADR-006: Markdown-Aware Chunking
**Decision**: Custom chunker preserving header hierarchy  
**Rationale**: 
- Headers become metadata for better citations
- Maintains document structure
- Enables filtering by section

## ADR-007: Multiple Interfaces
**Decision**: CLI + REST API + Web UI  
**Rationale**:
- CLI: Fast development/testing
- API: Production integration
- Web UI: Demos and non-technical users
