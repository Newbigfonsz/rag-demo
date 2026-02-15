"""RAG pipeline with caching, retry, fallback, and alerting."""

from dataclasses import dataclass
from typing import Generator as GenType, Optional
import time
from rich.console import Console

from src.config import settings
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.generation.generator import Generator, alert_manager
from src.analytics import QueryMetrics
from src.retrieval.query_rewriter import get_rewriter
from src.retrieval.cache import get_cache

@dataclass
class Message:
    role: str
    content: str

@dataclass 
class RAGResponse:
    query: str
    answer: str
    sources: list[RetrievedChunk]
    retrieval_scores: list[float]
    model: str
    reranked: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    query_id: int = None
    rewritten_query: Optional[str] = None
    cached: bool = False
    used_fallback: bool = False
    retries: int = 0

class RAGPipeline:
    def __init__(self, use_reranker: bool = True, llm_backend: str = "ollama", use_cache: bool = True):
        self.retriever = HybridRetriever()
        self.llm_backend = llm_backend
        self.generator = Generator(backend=llm_backend, auto_fallback=True, max_retries=3)
        self.console = Console()
        self.use_reranker = use_reranker
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
        self.reranker = None
        self.memory: list[Message] = []
        self.max_memory: int = 10
        self.query_rewriter = get_rewriter(llm_backend)
        
        if use_reranker:
            try:
                from src.retrieval.reranker import Reranker
                self.reranker = Reranker()
            except Exception as e:
                self.console.print(f"[yellow]Reranker not available: {e}[/yellow]")
                self.use_reranker = False
    
    def query(self, question: str, top_k: int = None, use_memory: bool = True, show_sources: bool = True, rewrite_query: bool = True, stream: bool = False) -> RAGResponse:
        start_time = time.time()
        top_k = top_k or settings.rerank_top_k
        cached = False
        used_fallback = False
        retries = 0
        
        # Query rewriting
        rewritten_query = None
        search_query = question
        if rewrite_query and use_memory and self.memory:
            rewritten = self.query_rewriter.rewrite(question, self.memory)
            if rewritten != question:
                rewritten_query = rewritten
                search_query = rewritten
        
        # Check cache first
        if self.use_cache and self.cache and not use_memory:
            cached_response = self.cache.get(search_query)
            if cached_response:
                latency_ms = (time.time() - start_time) * 1000
                return RAGResponse(
                    query=question,
                    answer=cached_response.answer,
                    sources=[],
                    retrieval_scores=[],
                    model=cached_response.model + " (cached)",
                    reranked=False,
                    latency_ms=latency_ms,
                    cached=True,
                    rewritten_query=rewritten_query
                )
        
        # Retrieve
        retrieve_k = top_k * 3 if self.use_reranker else top_k
        chunks = self.retriever.vector_search(search_query, top_k=retrieve_k)
        
        if not chunks:
            alert_manager.send_alert("warning", "No chunks retrieved", {"query": question[:50]})
        
        # Rerank
        reranked = False
        if self.use_reranker and self.reranker and len(chunks) > top_k:
            chunks = self.reranker.rerank(search_query, chunks, top_k=top_k)
            reranked = True
        else:
            chunks = chunks[:top_k]
        
        # Memory context
        memory_context = self._format_memory() if use_memory and self.memory else ""
        
        # Generate (with retry and fallback)
        generated = self.generator.generate(question, chunks, memory_context=memory_context)
        used_fallback = generated.used_fallback
        retries = generated.retries
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Cache the response
        if self.use_cache and self.cache and not use_memory and not cached:
            self.cache.set(
                search_query,
                generated.answer,
                [c.citation for c in chunks],
                generated.model
            )
        
        # Log to analytics
        metrics = QueryMetrics(
            query=question,
            answer=generated.answer,
            model=generated.model,
            llm_backend=self.llm_backend if not used_fallback else "claude (fallback)",
            retrieval_scores=[c.score for c in chunks],
            sources=[c.citation for c in chunks],
            latency_ms=latency_ms,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            cost_usd=generated.cost_usd,
            reranked=reranked,
        )
        query_id = metrics.log()
        
        # Update memory
        if use_memory:
            self.memory.append(Message(role="user", content=question))
            self.memory.append(Message(role="assistant", content=generated.answer))
            if len(self.memory) > self.max_memory * 2:
                self.memory = self.memory[-self.max_memory * 2:]
        
        return RAGResponse(
            query=question,
            answer=generated.answer,
            sources=chunks if show_sources else [],
            retrieval_scores=[c.score for c in chunks],
            model=generated.model,
            reranked=reranked,
            latency_ms=latency_ms,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            cost_usd=generated.cost_usd,
            query_id=query_id,
            rewritten_query=rewritten_query,
            cached=cached,
            used_fallback=used_fallback,
            retries=retries
        )
    
    def _format_memory(self) -> str:
        if not self.memory:
            return ""
        lines = ["Previous conversation:"]
        for msg in self.memory[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content[:200]}")
        return "\n".join(lines)
    
    def clear_memory(self):
        self.memory = []
    
    def switch_llm(self, backend: str):
        self.llm_backend = backend
        self.generator = Generator(backend=backend, auto_fallback=True)
    
    def get_cache_stats(self) -> dict:
        if self.cache:
            return self.cache.get_stats()
        return {}

def create_pipeline(use_reranker: bool = True, llm_backend: str = "ollama") -> RAGPipeline:
    return RAGPipeline(use_reranker=use_reranker, llm_backend=llm_backend)
