"""CLI with memory support."""

import sys
from rich.console import Console
from rich.panel import Panel
from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available

console = Console()

def run_chat():
    console.print(Panel.fit(
        "[bold magenta]Mystic RAG Chat[/bold magenta]\n\n"
        "Commands: quit, clear, memory, sources, debug",
        title="Welcome", border_style="magenta"
    ))
    
    if not check_llm_available():
        console.print("[red]LLM not available[/red]")
        sys.exit(1)
    
    console.print("[dim]Loading (reranker may take a moment)...[/dim]")
    pipeline = RAGPipeline(use_reranker=True)
    console.print("[green]Ready![/green]")
    
    show_sources, debug_mode = True, False
    
    while True:
        try:
            mem = f" [dim]({len(pipeline.memory)//2} msgs)[/dim]" if pipeline.memory else ""
            question = console.input(f"\n[bold cyan]You{mem}:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not question: continue
        
        # Commands
        if question.lower() == 'quit':
            console.print("[dim]Goodbye![/dim]")
            break
        if question.lower() == 'clear':
            pipeline.clear_memory()
            console.print("[dim]Memory cleared[/dim]")
            continue
        if question.lower() == 'memory':
            if pipeline.memory:
                console.print(f"[dim]Memory: {len(pipeline.memory)//2} exchanges[/dim]")
                for msg in pipeline.memory[-6:]:
                    role = "You" if msg.role == "user" else "Bot"
                    console.print(f"[dim]  {role}: {msg.content[:50]}...[/dim]")
            else:
                console.print("[dim]Memory is empty[/dim]")
            continue
        if question.lower() == 'sources':
            show_sources = not show_sources
            console.print(f"[dim]Sources {'on' if show_sources else 'off'}[/dim]")
            continue
        if question.lower() == 'debug':
            debug_mode = not debug_mode
            console.print(f"[dim]Debug {'on' if debug_mode else 'off'}[/dim]")
            continue
        
        # Process query
        try:
            with console.status("[green]Thinking...[/green]"):
                response = pipeline.query(question, use_memory=True)
            
            console.print(f"\n[bold green]Assistant:[/bold green]\n{response.answer}")
            
            if show_sources and response.sources:
                console.print("\n[dim]Sources:[/dim]")
                for i, s in enumerate(response.sources, 1):
                    console.print(f"[dim]  {i}. {s.citation} ({response.retrieval_scores[i-1]:.2f})[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    run_chat()
