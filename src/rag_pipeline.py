"""
Complete RAG pipeline combining retrieval and generation.
"""

from dataclasses import dataclass
from typing import Optional, Generator as GenType
from rich.console import Console
from rich.panel import Panel

from src.config import settings
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.generation.generator import Generator, GeneratedAnswer


@dataclass 
class RAGResponse:
    """Complete RAG response."""
    query: str
    answer: str
    sources: list[RetrievedChunk]
    retrieval_scores: list[float]
    model: str


class RAGPipeline:
    """End-to-end RAG pipeline."""
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = Generator()
        self.console = Console()
    
    def query(
        self,
        question: str,
        top_k: int = None,
        show_sources: bool = True,
    ) -> RAGResponse:
        """Process a question through the RAG pipeline."""
        top_k = top_k or settings.rerank_top_k
        
        chunks = self.retriever.vector_search(question, top_k=top_k)
        generated = self.generator.generate(question, chunks)
        
        return RAGResponse(
            query=question,
            answer=generated.answer,
            sources=chunks if show_sources else [],
            retrieval_scores=[c.score for c in chunks],
            model=generated.model,
        )
    
    def query_streaming(self, question: str, top_k: int = None) -> GenType[str, None, None]:
        """Process question with streaming response."""
        top_k = top_k or settings.rerank_top_k
        chunks = self.retriever.vector_search(question, top_k=top_k)
        
        for token in self.generator.generate_streaming(question, chunks):
            yield token
