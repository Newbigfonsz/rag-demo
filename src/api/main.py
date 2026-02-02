"""FastAPI REST API for Mystic RAG."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from src.rag_pipeline import RAGPipeline
from src.retrieval.retriever import HybridRetriever
from src.generation.generator import check_llm_available
from src.config import settings

app = FastAPI(
    title="Mystic RAG API",
    description="RAG API for Chinese Zodiac and Numerology",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[RAGPipeline] = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True


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


@app.get("/")
async def root():
    return {"name": "Mystic RAG API", "docs": "/docs"}


@app.get("/health")
async def health():
    qdrant_ok = False
    try:
        HybridRetriever().client.get_collections()
        qdrant_ok = True
    except: pass
    
    ollama_ok = check_llm_available()
    
    return {
        "status": "healthy" if (qdrant_ok and ollama_ok) else "degraded",
        "ollama": ollama_ok,
        "qdrant": qdrant_ok,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    
    try:
        pipeline = get_pipeline()
        response = pipeline.query(request.question, top_k=request.top_k)
        
        sources = []
        if request.include_sources:
            for i, chunk in enumerate(response.sources):
                sources.append(Source(
                    content=chunk.content[:400] + "...",
                    citation=chunk.citation,
                    score=response.retrieval_scores[i],
                ))
        
        return ChatResponse(
            answer=response.answer,
            sources=sources,
            query=request.question,
            model=response.model,
            latency_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as e:
        raise HTTPException(500, str(e))
