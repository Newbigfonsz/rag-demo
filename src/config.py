"""
Configuration management for Mystic RAG.
Uses pydantic-settings for environment variable loading.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent)
    data_raw_dir: Path = Field(default=None)
    data_processed_dir: Path = Field(default=None)
    
    # Ollama Configuration
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    embedding_model: str = Field(default="bge-m3")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="mystic_rag")
    
    # Chunking Configuration
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    
    # Retrieval Configuration
    top_k: int = Field(default=10)
    rerank_top_k: int = Field(default=5)
    hybrid_alpha: float = Field(default=0.7)  # Weight for vector vs BM25
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    # Logging
    log_level: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set derived paths after initialization
        if self.data_raw_dir is None:
            self.data_raw_dir = self.project_root / "data" / "raw"
        if self.data_processed_dir is None:
            self.data_processed_dir = self.project_root / "data" / "processed"


# Global settings instance
settings = Settings()
