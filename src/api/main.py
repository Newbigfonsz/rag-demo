"""FastAPI with health checks, cache stats, and alerts."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from qdrant_client import QdrantClient

from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available, alert_manager
from src.retrieval.cache import get_cache

app = FastAPI(title="Mystic RAG API", version="2.0")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_memory: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str
    latency_ms: float
    cost_usd: float
    cached: bool = False
    used_fallback: bool = False

pipeline = None

@app.on_event("startup")
def startup():
    global pipeline
    pipeline = RAGPipeline(use_reranker=True, llm_backend="ollama", use_cache=True)

@app.get("/health")
def health_check():
    """Full system health check."""
    status = {"status": "healthy", "components": {}}
    
    try:
        client = QdrantClient(host="qdrant", port=6333)
        client.get_collections()
        status["components"]["qdrant"] = "ok"
    except Exception as e:
        status["components"]["qdrant"] = f"error: {e}"
        status["status"] = "degraded"
    
    try:
        ollama.list()
        status["components"]["ollama"] = "ok"
    except Exception as e:
        status["components"]["ollama"] = f"error: {e}"
        status["status"] = "degraded"
    
    if check_llm_available("claude"):
        status["components"]["claude_fallback"] = "ready"
    else:
        status["components"]["claude_fallback"] = "not configured"
    
    return status

@app.get("/health/quick")
def quick_health():
    return {"status": "ok"}

@app.get("/stats")
def get_stats():
    """Get cache and alert stats."""
    cache = get_cache()
    return {
        "cache": cache.get_stats(),
        "recent_alerts": alert_manager.get_recent_alerts(5)
    }

@app.post("/cache/clear")
def clear_cache():
    """Clear expired cache entries."""
    cache = get_cache()
    cleared = cache.clear_expired()
    return {"cleared": cleared}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    try:
        response = pipeline.query(
            request.question,
            top_k=request.top_k,
            use_memory=request.use_memory
        )
        return QueryResponse(
            answer=response.answer,
            sources=[s.citation for s in response.sources],
            model=response.model,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            cached=response.cached,
            used_fallback=response.used_fallback
        )
    except Exception as e:
        alert_manager.send_alert("error", f"Query failed: {e}", {"question": request.question[:50]})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Mystic RAG API v2.0", "features": ["retry", "fallback", "cache", "alerts"]}
