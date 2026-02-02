"""
Main ingestion pipeline.

Orchestrates: Chunking -> Embedding -> Indexing

Usage:
    python -m src.ingestion.main
    
    # Or with make:
    make ingest
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import settings
from src.ingestion.chunker import MarkdownChunker, Chunk
from src.ingestion.embedder import OllamaEmbedder, check_ollama_available
from src.ingestion.indexer import QdrantIndexer, check_qdrant_available


console = Console()


def run_ingestion(
    input_dir: Path = None,
    recreate_collection: bool = True,
    save_chunks: bool = True,
) -> dict:
    """
    Run the full ingestion pipeline.
    
    Args:
        input_dir: Directory containing markdown files
        recreate_collection: Whether to recreate the vector collection
        save_chunks: Whether to save chunks to JSON for inspection
        
    Returns:
        Stats dict with ingestion results
    """
    input_dir = input_dir or settings.data_raw_dir
    
    console.print(Panel.fit(
        "[bold blue]Mystic RAG Ingestion Pipeline[/bold blue]\n"
        f"Input: {input_dir}\n"
        f"Collection: {settings.qdrant_collection}",
        title="Starting Ingestion"
    ))
    
    stats = {
        "start_time": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "files_processed": 0,
        "chunks_created": 0,
        "vectors_indexed": 0,
        "errors": [],
    }
    
    # Step 1: Check dependencies
    console.print("\n[bold]Step 1/4:[/bold] Checking dependencies...")
    
    if not check_ollama_available(settings.embedding_model):
        stats["errors"].append(f"Ollama model {settings.embedding_model} not available")
        console.print("[red]Ollama check failed[/red]")
        return stats
    console.print(f"  Ollama ready (model: {settings.embedding_model})")
    
    if not check_qdrant_available():
        stats["errors"].append("Qdrant not available")
        console.print("[red]Qdrant check failed[/red]")
        return stats
    console.print(f"  Qdrant ready ({settings.qdrant_host}:{settings.qdrant_port})")
    
    # Step 2: Chunk documents
    console.print("\n[bold]Step 2/4:[/bold] Chunking documents...")
    
    chunker = MarkdownChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    chunks = []
    files_by_category = {}
    
    for file_path in input_dir.rglob("*.md"):
        file_chunks = chunker.chunk_file(file_path)
        chunks.extend(file_chunks)
        
        # Track by category (parent folder)
        category = file_path.parent.name
        if category not in files_by_category:
            files_by_category[category] = 0
        files_by_category[category] += 1
        stats["files_processed"] += 1
    
    stats["chunks_created"] = len(chunks)
    
    # Display chunking results
    table = Table(title="Documents Processed")
    table.add_column("Category", style="cyan")
    table.add_column("Files", justify="right")
    
    for category, count in files_by_category.items():
        table.add_row(category, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{stats['files_processed']}[/bold]")
    
    console.print(table)
    console.print(f"  Created {len(chunks)} chunks")
    
    # Optionally save chunks for inspection
    if save_chunks:
        chunks_file = settings.data_processed_dir / "chunks.json"
        chunks_data = [
            {
                "content": c.content,
                "metadata": c.metadata,
                "citation": c.citation,
            }
            for c in chunks
        ]
        chunks_file.write_text(json.dumps(chunks_data, indent=2))
        console.print(f"  Saved chunks to {chunks_file}")
    
    # Step 3: Generate embeddings
    console.print("\n[bold]Step 3/4:[/bold] Generating embeddings...")
    
    embedder = OllamaEmbedder(model=settings.embedding_model)
    records = embedder.embed_chunks(chunks, show_progress=True)
    
    embedding_dim = len(records[0]["embedding"])
    console.print(f"  Generated {len(records)} embeddings (dim={embedding_dim})")
    
    # Step 4: Index in Qdrant
    console.print("\n[bold]Step 4/4:[/bold] Indexing in Qdrant...")
    
    indexer = QdrantIndexer()
    indexer.create_collection(
        embedding_dim=embedding_dim,
        recreate=recreate_collection,
    )
    
    indexed_count = indexer.index_records(records)
    stats["vectors_indexed"] = indexed_count
    
    console.print(f"  Indexed {indexed_count} vectors")
    
    # Final stats
    stats["end_time"] = datetime.now().isoformat()
    
    # Save stats
    stats_file = settings.data_processed_dir / "ingestion_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2))
    
    # Display summary
    console.print(Panel.fit(
        f"[green]Files processed: {stats['files_processed']}[/green]\n"
        f"[green]Chunks created: {stats['chunks_created']}[/green]\n"
        f"[green]Vectors indexed: {stats['vectors_indexed']}[/green]\n"
        f"\nCollection: [cyan]{settings.qdrant_collection}[/cyan]",
        title="Ingestion Complete"
    ))
    
    return stats


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory (default: data/raw)",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Don't recreate collection if it exists",
    )
    parser.add_argument(
        "--no-save-chunks",
        action="store_true",
        help="Don't save chunks JSON file",
    )
    
    args = parser.parse_args()
    
    stats = run_ingestion(
        input_dir=args.input_dir,
        recreate_collection=not args.no_recreate,
        save_chunks=not args.no_save_chunks,
    )
    
    if stats["errors"]:
        console.print(f"\n[red]Errors: {stats['errors']}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
