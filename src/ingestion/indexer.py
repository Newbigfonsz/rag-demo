"""
Vector database indexing using Qdrant.

Handles:
- Collection creation with proper schema
- Vector insertion with metadata
- BM25 index for hybrid search
"""

import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from src.config import settings


class QdrantIndexer:
    """
    Index vectors and text in Qdrant for hybrid search.
    
    Design decision (see docs/decisions.md ADR-002):
    - Qdrant chosen for native hybrid search support
    - Local Docker deployment for demo reliability
    - REST API enables easy debugging
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
    ):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        
        self.client = QdrantClient(host=self.host, port=self.port)
    
    def create_collection(
        self,
        embedding_dim: int,
        recreate: bool = False,
    ) -> None:
        """
        Create collection with vector and text indexing.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            recreate: If True, delete existing collection first
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists:
            if recreate:
                self.client.delete_collection(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
                return
        
        # Create collection with vector config
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        
        # Create payload index for text search (BM25)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="content",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=15,
                lowercase=True,
            ),
        )
        
        # Create index for metadata filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
        print(f"Created collection: {self.collection_name}")
        print(f"  - Vector dim: {embedding_dim}")
        print(f"  - Text index: enabled (BM25)")
        print(f"  - Metadata index: enabled")
    
    def index_records(
        self,
        records: list[dict],
        batch_size: int = 100,
    ) -> int:
        """
        Index records into Qdrant.
        
        Args:
            records: List of dicts with 'content', 'embedding', 'metadata'
            batch_size: Number of records per batch
            
        Returns:
            Number of records indexed
        """
        points = []
        
        for record in records:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=record["embedding"],
                payload={
                    "content": record["content"],
                    **record["metadata"],
                },
            )
            points.append(point)
        
        # Insert in batches
        total_indexed = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_indexed += len(batch)
        
        return total_indexed
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")


def check_qdrant_available(host: str = None, port: int = None) -> bool:
    """Check if Qdrant is running and accessible."""
    host = host or settings.qdrant_host
    port = port or settings.qdrant_port
    
    try:
        client = QdrantClient(host=host, port=port)
        # Try to list collections
        client.get_collections()
        return True
    except Exception as e:
        print(f"Qdrant not available at {host}:{port}")
        print(f"Error: {e}")
        print("\nStart Qdrant with:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        return False


if __name__ == "__main__":
    # Quick test
    if check_qdrant_available():
        indexer = QdrantIndexer()
        info = indexer.get_collection_info()
        print(f"Collection info: {info}")
