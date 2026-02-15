"""FastAPI with health checks."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from qdrant_client import QdrantClient

from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available

app = FastAPI(title="Mystic RAG API", version="2.0")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_memory: bool = False
    llm_backend: str = "ollama"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model: str
    latency_ms: float
    cost_usd: float

pipeline = None

@app.on_event("startup")
def startup():
    global pipeline
    pipeline = RAGPipeline(use_reranker=True, llm_backend="ollama")

@app.get("/health")
def health_check():
    """Full system health check."""
    status = {"status": "healthy", "components": {}}
    
    # Check Qdrant
    try:
        client = QdrantClient(host="qdrant", port=6333)
        client.get_collections()
        status["components"]["qdrant"] = "ok"
    except Exception as e:
        status["components"]["qdrant"] = f"error: {e}"
        status["status"] = "degraded"
    
    # Check Ollama
    try:
        ollama.list()
        status["components"]["ollama"] = "ok"
    except Exception as e:
        status["components"]["ollama"] = f"error: {e}"
        status["status"] = "degraded"
    
    # Check Claude (optional)
    if check_llm_available("claude"):
        status["components"]["claude"] = "ok"
    else:
        status["components"]["claude"] = "not configured"
    
    return status

@app.get("/health/quick")
def quick_health():
    """Quick health check for load balancers."""
    return {"status": "ok"}

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
            cost_usd=response.cost_usd
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Mystic RAG API", "docs": "/docs"}
