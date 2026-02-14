"""CLI with LLM selection (Ollama or Claude)."""

import sys
import os
from rich.console import Console
from rich.panel import Panel
from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available

console = Console()

def run_chat():
    console.print(Panel.fit(
        "[bold magenta]Mystic RAG Chat[/bold magenta]\n\n"
        "Commands:\n"
        "  quit    - Exit\n"
        "  clear   - Clear memory\n"
        "  memory  - Show history\n"
        "  llm     - Switch LLM (ollama/claude)\n"
        "  sources - Toggle sources\n"
        "  debug   - Toggle debug",
        title="Welcome", border_style="magenta"
    ))
    
    # Check which LLMs are available
    ollama_ok = check_llm_available("ollama")
    claude_ok = check_llm_available("claude")
    
    console.print(f"\n[dim]LLM Status:[/dim]")
    console.print(f"  Ollama: {'[green]✓[/green]' if ollama_ok else '[red]✗[/red]'}")
    console.print(f"  Claude: {'[green]✓[/green]' if claude_ok else '[yellow]✗ (set ANTHROPIC_API_KEY)[/yellow]'}")
    
    # Default to ollama, fallback to claude
    if ollama_ok:
        llm_backend = "ollama"
    elif claude_ok:
        llm_backend = "claude"
    else:
        console.print("[red]No LLM available![/red]")
        sys.exit(1)
    
    console.print(f"\n[dim]Loading pipeline with {llm_backend}...[/dim]")
    pipeline = RAGPipeline(use_reranker=True, llm_backend=llm_backend)
    console.print(f"[green]Ready! Using {llm_backend}[/green]")
    
    show_sources, debug_mode = True, False
    
    while True:
        try:
            mem = f" [dim]({len(pipeline.memory)//2} msgs)[/dim]" if pipeline.memory else ""
            llm_tag = f"[dim][{pipeline.llm_backend}][/dim]"
            question = console.input(f"\n[bold cyan]You{mem} {llm_tag}:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
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
        
        if question.lower() == 'llm':
            current = pipeline.llm_backend
            new_backend = "claude" if current == "ollama" else "ollama"
            
            if check_llm_available(new_backend):
                pipeline.switch_llm(new_backend)
                console.print(f"[green]Switched to {new_backend}[/green]")
            else:
                console.print(f"[red]{new_backend} not available[/red]")
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
            with console.status(f"[green]Thinking ({pipeline.llm_backend})...[/green]"):
                response = pipeline.query(question, use_memory=True)
            
            console.print(f"\n[bold green]Assistant [{response.model}]:[/bold green]\n{response.answer}")
            
            if show_sources and response.sources:
                console.print("\n[dim]Sources:[/dim]")
                for i, s in enumerate(response.sources, 1):
                    console.print(f"[dim]  {i}. {s.citation} ({response.retrieval_scores[i-1]:.2f})[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    run_chat()
