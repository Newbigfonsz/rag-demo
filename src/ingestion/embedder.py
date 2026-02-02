"""
Embedding generation using Ollama.
"""

import ollama
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ingestion.chunker import Chunk


class OllamaEmbedder:
    """Generate embeddings using Ollama's local models."""
    
    def __init__(self, model: str = "bge-m3"):
        self.model = model
        self._embedding_dim = None
        
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dim is None:
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
        """Generate embeddings for chunks and return prepared records."""
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
    """Convenience function to embed chunks."""
    embedder = OllamaEmbedder(model=model)
    return embedder.embed_chunks(chunks, show_progress=show_progress)


def check_ollama_available(model: str = "bge-m3") -> bool:
    """Check if Ollama is running and model is available."""
    try:
        # Try to generate a test embedding - most reliable check
        response = ollama.embeddings(model=model, prompt="test")
        if "embedding" in response:
            return True
        return False
    except Exception as e:
        print(f"Ollama not available: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"And model is pulled: ollama pull {model}")
        return False


if __name__ == "__main__":
    if check_ollama_available():
        embedder = OllamaEmbedder()
        test_texts = ["The Dragon is a powerful zodiac sign"]
        embeddings = embedder.embed_texts(test_texts)
        print(f"Embedding dimension: {len(embeddings[0])}")
