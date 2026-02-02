"""
Embedding generation using Ollama.

Uses BGE-M3 model for multilingual embeddings that work well
for domain-specific content like our numerology/zodiac corpus.
"""

import ollama
from typing import Union
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.chunker import Chunk


class OllamaEmbedder:
    """
    Generate embeddings using Ollama's local models.
    
    Design decision (see docs/decisions.md ADR-001):
    - BGE-M3 chosen for good quality and local execution
    - No API costs or rate limits
    - Reliable for demos (no network dependency)
    """
    
    def __init__(self, model: str = "bge-m3"):
        self.model = model
        self._embedding_dim = None
        
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dim is None:
            # Generate a test embedding to get dimensions
            test_embedding = self.embed_text("test")
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = ollama.embeddings(
            model=self.model,
            prompt=text,
        )
        return response["embedding"]
    
    def embed_texts(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Generating embeddings ({len(texts)} texts)...",
                    total=len(texts)
                )
                
                for text in texts:
                    embedding = self.embed_text(text)
                    embeddings.append(embedding)
                    progress.advance(task)
        else:
            for text in texts:
                embeddings.append(self.embed_text(text))
        
        return embeddings
    
    def embed_chunks(self, chunks: list[Chunk], show_progress: bool = True) -> list[dict]:
        """
        Generate embeddings for chunks and return prepared records.
        
        Returns list of dicts ready for vector DB insertion:
        {
            "content": str,
            "embedding": list[float],
            "metadata": dict
        }
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "content": chunk.content,
                "embedding": embedding,
                "metadata": chunk.metadata,
            })
        
        return records


def embed_chunks(
    chunks: list[Chunk],
    model: str = "bge-m3",
    show_progress: bool = True,
) -> list[dict]:
    """
    Convenience function to embed chunks.
    
    Args:
        chunks: List of Chunk objects
        model: Ollama embedding model name
        show_progress: Whether to show progress bar
        
    Returns:
        List of records with embeddings and metadata
    """
    embedder = OllamaEmbedder(model=model)
    return embedder.embed_chunks(chunks, show_progress=show_progress)


def check_ollama_available(model: str = "bge-m3") -> bool:
    """Check if Ollama is running and model is available."""
    try:
        # Try to list models
        models = ollama.list()
        model_names = [m["name"].split(":")[0] for m in models.get("models", [])]
        
        if model not in model_names:
            print(f"Model '{model}' not found. Available models: {model_names}")
            print(f"Run: ollama pull {model}")
            return False
        
        return True
    except Exception as e:
        print(f"Ollama not available: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False


if __name__ == "__main__":
    # Quick test
    if check_ollama_available():
        embedder = OllamaEmbedder()
        
        test_texts = [
            "The Dragon is a powerful zodiac sign",
            "Life Path 7 represents wisdom and introspection",
            "Rats are known for their intelligence",
        ]
        
        embeddings = embedder.embed_texts(test_texts)
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
