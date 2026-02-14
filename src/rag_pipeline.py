"""RAG pipeline with reranking, memory, and multi-LLM support."""

from dataclasses import dataclass
from typing import Generator as GenType
from rich.console import Console

from src.config import settings
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.generation.generator import Generator


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


class RAGPipeline:
    """RAG pipeline with reranking, memory, and LLM selection."""
    
    def __init__(self, use_reranker: bool = True, llm_backend: str = "ollama"):
        """
        Initialize pipeline.
        
        Args:
            use_reranker: Enable cross-encoder reranking
            llm_backend: "ollama" (local, free) or "claude" (API, better)
        """
        self.retriever = HybridRetriever()
        self.llm_backend = llm_backend
        self.generator = Generator(backend=llm_backend)
        self.console = Console()
        self.use_reranker = use_reranker
        self.reranker = None
        self.memory: list[Message] = []
        self.max_memory: int = 10
        
        if use_reranker:
            try:
                from src.retrieval.reranker import Reranker
                self.reranker = Reranker()
            except Exception as e:
                self.console.print(f"[yellow]Reranker not available: {e}[/yellow]")
                self.use_reranker = False
    
    def query(self, question: str, top_k: int = None, use_memory: bool = True, show_sources: bool = True) -> RAGResponse:
        top_k = top_k or settings.rerank_top_k
        
        retrieve_k = top_k * 3 if self.use_reranker else top_k
        chunks = self.retriever.vector_search(question, top_k=retrieve_k)
        
        reranked = False
        if self.use_reranker and self.reranker and len(chunks) > top_k:
            chunks = self.reranker.rerank(question, chunks, top_k=top_k)
            reranked = True
        else:
            chunks = chunks[:top_k]
        
        memory_context = self._format_memory() if use_memory and self.memory else ""
        
        generated = self.generator.generate(question, chunks, memory_context=memory_context)
        
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
        """Switch LLM backend (ollama or claude)."""
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
