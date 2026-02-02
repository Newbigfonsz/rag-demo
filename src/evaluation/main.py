"""Evaluation pipeline for Mystic RAG."""

import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import settings
from src.rag_pipeline import RAGPipeline


@dataclass
class EvalResult:
    query: str
    retrieved_sources: list
    top_score: float
    latency_ms: float


TEST_CASES = [
    "What are the personality traits of a Rat?",
    "Are Dragon and Monkey compatible?",
    "What careers suit Life Path 7?",
    "Tell me about the Wood Tiger",
    "What is a Master Number?",
    "What are lucky colors for the Ox?",
    "How do I calculate my life path number?",
    "Which signs are compatible with Rat?",
    "What element years exist for Snake?",
    "Difference between Life Path 1 and 8?",
]


class Evaluator:
    def __init__(self):
        self.pipeline = RAGPipeline(use_reranker=True)
        self.console = Console()
    
    def run(self):
        results = []
        
        self.console.print(Panel.fit(f"Running {len(TEST_CASES)} tests", title="Evaluation"))
        
        for i, query in enumerate(TEST_CASES, 1):
            self.console.print(f"\n[{i}/{len(TEST_CASES)}] {query[:40]}...")
            
            start = time.time()
            response = self.pipeline.query(query, top_k=5, use_memory=False)
            latency = (time.time() - start) * 1000
            
            retrieved = [s.source_file for s in response.sources]
            top_score = response.retrieval_scores[0] if response.retrieval_scores else -99
            
            results.append(EvalResult(
                query=query,
                retrieved_sources=retrieved,
                top_score=round(top_score, 3),
                latency_ms=round(latency, 1),
            ))
            
            # For reranker: positive = relevant, negative = irrelevant
            status = "✓" if top_score > 0 else "⚠"
            self.console.print(f"  {status} Top Score: {top_score:.3f}, Latency: {latency:.0f}ms")
        
        return results
    
    def print_summary(self, results):
        table = Table(title="Results")
        table.add_column("Query", width=35)
        table.add_column("Top Score", width=10)
        table.add_column("Latency", width=10)
        
        for r in results:
            style = "green" if r.top_score > 0 else "yellow" if r.top_score > -5 else "red"
            table.add_row(r.query[:35], f"[{style}]{r.top_score:.2f}[/{style}]", f"{r.latency_ms:.0f}ms")
        
        self.console.print(table)
        
        avg_score = sum(r.top_score for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        passed = sum(1 for r in results if r.top_score > 0)
        
        self.console.print(Panel.fit(
            f"Avg Top Score: {avg_score:.3f}\n"
            f"Avg Latency: {avg_latency:.0f}ms\n"
            f"Passed (score > 0): {passed}/{len(results)}",
            title="Summary"
        ))


def main():
    evaluator = Evaluator()
    results = evaluator.run()
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
