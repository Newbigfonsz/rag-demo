"""RAG pipeline with reranking, memory, query rewriting, and multi-LLM support."""

from dataclasses import dataclass
from typing import Generator as GenType, Optional
import time
from rich.console import Console

from src.config import settings
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.generation.generator import Generator
from src.analytics import QueryMetrics
from src.retrieval.query_rewriter import get_rewriter

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

class RAGPipeline:
    def __init__(self, use_reranker: bool = True, llm_backend: str = "ollama"):
        self.retriever = HybridRetriever()
        self.llm_backend = llm_backend
        self.generator = Generator(backend=llm_backend)
        self.console = Console()
        self.use_reranker = use_reranker
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
        rewritten_query = None
        search_query = question
        if rewrite_query and use_memory and self.memory:
            rewritten = self.query_rewriter.rewrite(question, self.memory)
            if rewritten != question:
                rewritten_query = rewritten
                search_query = rewritten
        retrieve_k = top_k * 3 if self.use_reranker else top_k
        chunks = self.retriever.vector_search(search_query, top_k=retrieve_k)
        reranked = False
        if self.use_reranker and self.reranker and len(chunks) > top_k:
            chunks = self.reranker.rerank(search_query, chunks, top_k=top_k)
            reranked = True
        else:
            chunks = chunks[:top_k]
        memory_context = self._format_memory() if use_memory and self.memory else ""
        generated = self.generator.generate(question, chunks, memory_context=memory_context)
        latency_ms = (time.time() - start_time) * 1000
        metrics = QueryMetrics(query=question, answer=generated.answer, model=generated.model, llm_backend=self.llm_backend, retrieval_scores=[c.score for c in chunks], sources=[c.citation for c in chunks], latency_ms=latency_ms, input_tokens=generated.input_tokens, output_tokens=generated.output_tokens, cost_usd=generated.cost_usd, reranked=reranked)
        query_id = metrics.log()
        if use_memory:
            self.memory.append(Message(role="user", content=question))
            self.memory.append(Message(role="assistant", content=generated.answer))
            if len(self.memory) > self.max_memory * 2:
                self.memory = self.memory[-self.max_memory * 2:]
        return RAGResponse(query=question, answer=generated.answer, sources=chunks if show_sources else [], retrieval_scores=[c.score for c in chunks], model=generated.model, reranked=reranked, latency_ms=latency_ms, input_tokens=generated.input_tokens, output_tokens=generated.output_tokens, cost_usd=generated.cost_usd, query_id=query_id, rewritten_query=rewritten_query)
    
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
        self.generator = Generator(backend=backend)
    
    def query_streaming(self, question: str, top_k: int = None) -> GenType[str, None, None]:
        top_k = top_k or settings.rerank_top_k
        retrieve_k = top_k * 3 if self.use_reranker else top_k
        chunks = self.retriever.vector_search(question, top_k=retrieve_k)
        if self.use_reranker and self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_k=top_k)
        else:
            chunks = chunks[:top_k]
        for token in self.generator.generate_streaming(question, chunks):
            yield token

def create_pipeline(use_reranker: bool = True, llm_backend: str = "ollama") -> RAGPipeline:
    return RAGPipeline(use_reranker=use_reranker, llm_backend=llm_backend)
