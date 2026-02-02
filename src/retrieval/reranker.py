"""Cross-encoder reranking."""

from sentence_transformers import CrossEncoder
from src.retrieval.retriever import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not chunks or len(chunks) <= top_k:
            return chunks
        
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        return [
            RetrievedChunk(content=c.content, source_file=c.source_file, headers=c.headers, score=float(s))
            for c, s in scored[:top_k]
        ]


if __name__ == "__main__":
    from src.retrieval.retriever import HybridRetriever
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    console.print("[bold]Loading reranker...[/bold]")
    reranker = Reranker()
    
    retriever = HybridRetriever()
    query = "What careers suit Life Path 7?"
    
    console.print(f"\n[cyan]Query:[/cyan] {query}\n")
    
    initial = retriever.vector_search(query, top_k=10)
    reranked = reranker.rerank(query, initial, top_k=5)
    
    table = Table(title="Before Reranking")
    table.add_column("Rank"); table.add_column("Score"); table.add_column("Source")
    for i, c in enumerate(initial[:5], 1):
        table.add_row(str(i), f"{c.score:.3f}", c.citation[:40])
    console.print(table)
    
    table2 = Table(title="After Reranking")
    table2.add_column("Rank"); table2.add_column("Score"); table2.add_column("Source")
    for i, c in enumerate(reranked, 1):
        table2.add_row(str(i), f"{c.score:.3f}", c.citation[:40])
    console.print(table2)
