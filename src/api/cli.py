"""CLI with Agentic RAG and query rewriting."""

import sys
import os
from rich.console import Console
from rich.panel import Panel
from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available
from src.agents.rag_agent import RAGAgent

console = Console()

def run_chat():
    console.print(Panel.fit("[bold magenta]Mystic RAG Chat[/bold magenta]\n\nCommands: quit, clear, memory, llm, agent, sources", title="Welcome", border_style="magenta"))
    ollama_ok = check_llm_available("ollama")
    claude_ok = check_llm_available("claude")
    console.print(f"\n[dim]LLM: Ollama {'Y' if ollama_ok else 'N'} | Claude {'Y' if claude_ok else 'N'}[/dim]")
    llm_backend = "ollama" if ollama_ok else "claude" if claude_ok else None
    if not llm_backend:
        console.print("[red]No LLM available![/red]")
        sys.exit(1)
    console.print(f"[dim]Loading...[/dim]")
    pipeline = RAGPipeline(use_reranker=True, llm_backend=llm_backend)
    agent = RAGAgent(pipeline)
    console.print(f"[green]Ready! Using {llm_backend}[/green]")
    show_sources, agent_mode = True, True
    while True:
        try:
            mem = f" ({len(pipeline.memory)//2} msgs)" if pipeline.memory else ""
            mode = "A" if agent_mode else "Q"
            question = console.input(f"\n[bold cyan]You{mem} [{mode}]:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        if not question:
            continue
        cmd = question.lower()
        if cmd == "quit":
            break
        if cmd == "clear":
            pipeline.clear_memory()
            console.print("[dim]Memory cleared[/dim]")
            continue
        if cmd == "memory":
            if pipeline.memory:
                for msg in pipeline.memory[-6:]:
                    console.print(f"[dim]{msg.role}: {msg.content[:50]}...[/dim]")
            else:
                console.print("[dim]Empty[/dim]")
            continue
        if cmd == "llm":
            new = "claude" if pipeline.llm_backend == "ollama" else "ollama"
            if check_llm_available(new):
                pipeline.switch_llm(new)
                console.print(f"[green]Switched to {new}[/green]")
            continue
        if cmd == "agent":
            agent_mode = not agent_mode
            console.print(f"[dim]Agent {'ON' if agent_mode else 'OFF'}[/dim]")
            continue
        if cmd == "sources":
            show_sources = not show_sources
            continue
        try:
            with console.status("[green]Thinking...[/green]"):
                if agent_mode:
                    result = agent.run(question, use_memory=True)
                    response_text = result["response"]
                    action = result.get("action_taken", "search")
                    sources = result.get("sources", [])
                    rewritten = result.get("rewritten_query")
                    is_clarification = result.get("is_clarification", False)
                else:
                    response = pipeline.query(question, use_memory=True, rewrite_query=True)
                    response_text = response.answer
                    action = "search"
                    sources = [s.citation for s in response.sources]
                    rewritten = response.rewritten_query
                    is_clarification = False
            if rewritten:
                console.print(f"[dim]Rewritten: {rewritten}[/dim]")
            if agent_mode and action != "search":
                console.print(f"[dim]Action: {action}[/dim]")
            style = "[bold yellow]" if is_clarification else "[bold green]"
            console.print(f"\n{style}Assistant:[/]\n{response_text}")
            if show_sources and sources and not is_clarification:
                console.print("[dim]Sources: " + ", ".join(sources[:3]) + "[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    run_chat()
