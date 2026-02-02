"""
CLI chat interface for Mystic RAG.

Usage: python -m src.api.cli
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available

console = Console()


def print_welcome():
    console.print(Panel.fit(
        "[bold magenta]Mystic RAG Chat[/bold magenta]\n\n"
        "Ask about:\n"
        "  Chinese zodiac signs and compatibility\n"
        "  Numerology life path numbers\n\n"
        "[dim]Commands: quit, sources, debug, clear[/dim]",
        title="Welcome",
        border_style="magenta"
    ))
    
    examples = [
        "What does it mean to be a Dragon?",
        "Are Rat and Ox compatible?",
        "What careers suit Life Path 7?",
    ]
    console.print("\n[dim]Try asking:[/dim]")
    for ex in examples:
        console.print(f"[dim]  {ex}[/dim]")


def run_chat():
    print_welcome()
    
    console.print("\n[dim]Checking LLM...[/dim]")
    if not check_llm_available():
        console.print("[red]LLM not available. Run: ollama pull llama3.2[/red]")
        sys.exit(1)
    console.print("[green]Ready![/green]")
    
    pipeline = RAGPipeline()
    show_sources = True
    debug_mode = False
    
    while True:
        try:
            console.print()
            question = console.input("[bold cyan]You:[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        question = question.strip()
        if not question:
            continue
        
        if question.lower() == 'quit':
            console.print("[dim]Goodbye![/dim]")
            break
        
        if question.lower() == 'sources':
            show_sources = not show_sources
            console.print(f"[dim]Sources {'on' if show_sources else 'off'}[/dim]")
            continue
        
        if question.lower() == 'debug':
            debug_mode = not debug_mode
            console.print(f"[dim]Debug {'on' if debug_mode else 'off'}[/dim]")
            continue
        
        if question.lower() == 'clear':
            console.clear()
            print_welcome()
            continue
        
        try:
            with console.status("[green]Thinking...[/green]"):
                response = pipeline.query(question)
            
            if debug_mode and response.sources:
                table = Table(title="Retrieved Chunks", show_header=True)
                table.add_column("Score", width=8)
                table.add_column("Source", width=25)
                table.add_column("Preview", width=45)
                for i, s in enumerate(response.sources):
                    table.add_row(f"{response.retrieval_scores[i]:.3f}", s.citation, s.content[:60]+"...")
                console.print(table)
            
            console.print(f"\n[bold green]Assistant:[/bold green]")
            console.print(response.answer)
            
            if show_sources and response.sources:
                console.print("\n[dim]Sources:[/dim]")
                for i, s in enumerate(response.sources, 1):
                    console.print(f"[dim]  {i}. {s.citation} ({response.retrieval_scores[i-1]:.2f})[/dim]")
                    
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    run_chat()
