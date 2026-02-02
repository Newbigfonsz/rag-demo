"""
Ingestion pipeline for Mystic RAG.

Components:
- chunker: Markdown-aware document chunking
- embedder: Ollama-based embedding generation
- indexer: Qdrant vector storage
"""

from src.ingestion.chunker import Chunk, MarkdownChunker, chunk_documents
from src.ingestion.embedder import OllamaEmbedder, embed_chunks, check_ollama_available
from src.ingestion.indexer import QdrantIndexer, check_qdrant_available
from src.ingestion.main import run_ingestion

__all__ = [
    "Chunk",
    "MarkdownChunker",
    "chunk_documents",
    "OllamaEmbedder",
    "embed_chunks",
    "check_ollama_available",
    "QdrantIndexer",
    "check_qdrant_available",
    "run_ingestion",
]
