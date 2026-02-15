"""Semantic cache for similar queries."""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from typing import Optional

@dataclass
class CachedResponse:
    query: str
    answer: str
    sources: list
    model: str
    timestamp: float
    hit_count: int = 0

class SemanticCache:
    """Cache responses for similar queries."""
    
    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24, similarity_threshold: float = 0.92):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.similarity_threshold = similarity_threshold
        self.cache_file = self.cache_dir / "query_cache.json"
        self.embeddings_file = self.cache_dir / "query_embeddings.npy"
        self.cache = self._load_cache()
        self.embeddings = self._load_embeddings()
        self.stats = {"hits": 0, "misses": 0}
    
    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _load_embeddings(self) -> dict:
        if self.embeddings_file.exists():
            try:
                data = np.load(self.embeddings_file, allow_pickle=True).item()
                return data if isinstance(data, dict) else {}
            except:
                return {}
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)
    
    def _save_embeddings(self):
        np.save(self.embeddings_file, self.embeddings)
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get(self, query: str, query_embedding: list = None) -> Optional[CachedResponse]:
        """Get cached response for exact or similar query."""
        query_hash = self._hash_query(query)
        current_time = time.time()
        
        # Exact match
        if query_hash in self.cache:
            cached = self.cache[query_hash]
            if current_time - cached["timestamp"] < self.ttl_seconds:
                self.stats["hits"] += 1
                cached["hit_count"] += 1
                self._save_cache()
                return CachedResponse(**cached)
        
        # Semantic similarity match
        if query_embedding and self.embeddings:
            for cached_hash, cached_embedding in self.embeddings.items():
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                if similarity >= self.similarity_threshold:
                    if cached_hash in self.cache:
                        cached = self.cache[cached_hash]
                        if current_time - cached["timestamp"] < self.ttl_seconds:
                            self.stats["hits"] += 1
                            cached["hit_count"] += 1
                            self._save_cache()
                            print(f"[CACHE] Semantic hit (similarity={similarity:.3f})")
                            return CachedResponse(**cached)
        
        self.stats["misses"] += 1
        return None
    
    def set(self, query: str, answer: str, sources: list, model: str, query_embedding: list = None):
        """Cache a response."""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "model": model,
            "timestamp": time.time(),
            "hit_count": 0
        }
        if query_embedding:
            self.embeddings[query_hash] = query_embedding
            self._save_embeddings()
        self._save_cache()
    
    def clear_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired = [k for k, v in self.cache.items() if current_time - v["timestamp"] > self.ttl_seconds]
        for k in expired:
            del self.cache[k]
            if k in self.embeddings:
                del self.embeddings[k]
        if expired:
            self._save_cache()
            self._save_embeddings()
        return len(expired)
    
    def get_stats(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "cached_queries": len(self.cache)
        }

# Singleton
_cache = None

def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
