"""Hybrid retriever with error handling."""

from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from src.ingestion.embedder import OllamaEmbedder

@dataclass
class RetrievedChunk:
    text: str
    citation: str
    score: float
    metadata: dict = None

class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(host="qdrant", port=6333)
        self.embedder = OllamaEmbedder(model="nomic-embed-text")
        self.collection = "mystic_rag"
    
    def vector_search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        try:
            query_embedding = self.embedder.embed_text(query)
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            return []
        
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128, exact=False)
            )
        except Exception as e:
            print(f"[ERROR] Qdrant search failed: {e}")
            return []
        
        chunks = []
        for r in results:
            chunks.append(RetrievedChunk(
                text=r.payload.get("text", ""),
                citation=f"{r.payload.get('category', 'unknown')}/{r.payload.get('source', 'unknown')}",
                score=r.score,
                metadata=r.payload
            ))
        return chunks
