"""FastAPI with reranking."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available

app = FastAPI(title="Mystic RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(use_reranker=True)
    return _pipeline


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_memory: bool = True


class Source(BaseModel):
    content: str
    citation: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    query: str
    model: str
    latency_ms: float
    reranked: bool


@app.get("/")
async def root():
    return {"name": "Mystic RAG API", "docs": "/docs", "features": ["reranking", "memory"]}


@app.get("/health")
async def health():
    return {"status": "healthy", "llm": check_llm_available()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    try:
        pipeline = get_pipeline()
        response = pipeline.query(request.question, top_k=request.top_k, use_memory=request.use_memory)
        
        return ChatResponse(
            answer=response.answer,
            sources=[Source(content=c.content[:400], citation=c.citation, score=response.retrieval_scores[i]) for i, c in enumerate(response.sources)],
            query=request.question,
            model=response.model,
            latency_ms=round((time.time() - start) * 1000, 2),
            reranked=response.reranked,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/clear-memory")
async def clear_memory():
    get_pipeline().clear_memory()
    return {"status": "memory cleared"}
