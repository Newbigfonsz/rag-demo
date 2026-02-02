"""
Hybrid retrieval combining vector search and BM25.
"""

from dataclasses import dataclass
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import settings
from src.ingestion.embedder import OllamaEmbedder


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score."""
    content: str
    source_file: str
    headers: str
    score: float
    
    @property
    def citation(self) -> str:
        """Format as citation string."""
        if self.headers:
            return f"{self.source_file}: {self.headers}"
        return self.source_file
    
    def __repr__(self) -> str:
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"RetrievedChunk(score={self.score:.3f}, source={self.source_file})"


class HybridRetriever:
    """
    Hybrid retrieval using vector similarity + BM25 keyword matching.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        host: str = None,
        port: int = None,
    ):
        self.collection_name = collection_name or settings.qdrant_collection
        self.embedding_model = embedding_model or settings.embedding_model
        
        self.client = QdrantClient(
            host=host or settings.qdrant_host,
            port=port or settings.qdrant_port,
        )
        self.embedder = OllamaEmbedder(model=self.embedding_model)
    
    def vector_search(
        self,
        query: str,
        top_k: int = None,
        filter_source: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Pure vector similarity search."""
        top_k = top_k or settings.top_k
        
        query_embedding = self.embedder.embed_text(query)
        
        query_filter = None
        if filter_source:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_file",
                        match=models.MatchValue(value=filter_source),
                    )
                ]
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        
        chunks = []
        for point in results:
            chunk = RetrievedChunk(
                content=point.payload.get("content", ""),
                source_file=point.payload.get("source_file", ""),
                headers=point.payload.get("headers", ""),
                score=point.score,
            )
            chunks.append(chunk)
        
        return chunks
    
    def keyword_search(
        self,
        query: str,
        top_k: int = None,
    ) -> list[RetrievedChunk]:
        """Pure BM25 keyword search."""
        top_k = top_k or settings.top_k
        
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="content",
                    match=models.MatchText(text=query),
                )
            ]
        )
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        
        chunks = []
        for point in results[0]:
            chunk = RetrievedChunk(
                content=point.payload.get("content", ""),
                source_file=point.payload.get("source_file", ""),
                headers=point.payload.get("headers", ""),
                score=1.0,
            )
            chunks.append(chunk)
        
        return chunks


def retrieve(
    query: str,
    top_k: int = None,
    method: str = "vector",
) -> list[RetrievedChunk]:
    """Convenience function for retrieval."""
    retriever = HybridRetriever()
    
    if method == "keyword":
        return retriever.keyword_search(query, top_k=top_k)
    else:
        return retriever.vector_search(query, top_k=top_k)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    retriever = HybridRetriever()
    
    query = "What are Dragons compatible with?"
    console.print(f"\n[bold]Query:[/bold] {query}")
    
    results = retriever.vector_search(query, top_k=3)
    
    table = Table(title="Results")
    table.add_column("Score", style="cyan", width=8)
    table.add_column("Source", style="green", width=25)
    table.add_column("Preview", width=50)
    
    for r in results:
        preview = r.content[:80] + "..."
        table.add_row(f"{r.score:.3f}", r.citation, preview)
    
    console.print(table)
