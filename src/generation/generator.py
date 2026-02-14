"""
LLM generation with Ollama and Claude support + token tracking.
"""

import os
from dataclasses import dataclass
from src.config import settings
from src.retrieval.retriever import RetrievedChunk


@dataclass
class GeneratedAnswer:
    answer: str
    sources: list[RetrievedChunk]
    query: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in Chinese zodiac signs and numerology.

RULES:
1. ONLY use information from the provided context
2. If the context doesn't have the answer, say "I don't have information about that"
3. Be specific about which zodiac sign or life path number you're discussing
4. If there's conversation history, use it to understand follow-up questions
5. Keep answers conversational but informative"""


class OllamaGenerator:
    def __init__(self, model: str = None):
        import ollama
        self.ollama = ollama
        self.model = model or settings.ollama_model
    
    def generate(self, query: str, context_chunks: list[RetrievedChunk], memory_context: str = "") -> GeneratedAnswer:
        context = self._format_context(context_chunks)
        
        if memory_context:
            prompt = f"{memory_context}\n\n<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer based on context and history."
        else:
            prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer based on the context."
        
        response = self.ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            options={"num_predict": 1024, "temperature": 0.7},
        )
        
        return GeneratedAnswer(
            answer=response["message"]["content"],
            sources=context_chunks,
            query=query,
            model=f"ollama/{self.model}",
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            cost_usd=0.0,  # Ollama is free
        )
    
    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n---\n\n".join([f"[Source {i}: {c.citation}]\n{c.content}" for i, c in enumerate(chunks, 1)])
    
    def generate_streaming(self, query: str, context_chunks: list[RetrievedChunk], temperature: float = 0.7):
        context = self._format_context(context_chunks)
        prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer:"
        stream = self.ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            options={"temperature": temperature},
            stream=True,
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]


class ClaudeGenerator:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
    
    def generate(self, query: str, context_chunks: list[RetrievedChunk], memory_context: str = "") -> GeneratedAnswer:
        from src.analytics import calculate_claude_cost
        
        context = self._format_context(context_chunks)
        
        if memory_context:
            prompt = f"{memory_context}\n\n<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer based on context and history."
        else:
            prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer based on the context."
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_claude_cost(self.model, input_tokens, output_tokens)
        
        return GeneratedAnswer(
            answer=response.content[0].text,
            sources=context_chunks,
            query=query,
            model=f"claude/{self.model}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
    
    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n---\n\n".join([f"[Source {i}: {c.citation}]\n{c.content}" for i, c in enumerate(chunks, 1)])
    
    def generate_streaming(self, query: str, context_chunks: list[RetrievedChunk], temperature: float = 0.7):
        context = self._format_context(context_chunks)
        prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer:"
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


class Generator:
    def __init__(self, backend: str = "ollama", model: str = None):
        self.backend = backend
        
        if backend == "claude":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY required for Claude")
            self._generator = ClaudeGenerator(model or "claude-sonnet-4-20250514")
        else:
            self._generator = OllamaGenerator(model or settings.ollama_model)
    
    def generate(self, query: str, context_chunks: list[RetrievedChunk], memory_context: str = "", **kwargs) -> GeneratedAnswer:
        return self._generator.generate(query, context_chunks, memory_context=memory_context)
    
    def generate_streaming(self, query: str, context_chunks: list[RetrievedChunk], **kwargs):
        return self._generator.generate_streaming(query, context_chunks, **kwargs)


def check_llm_available(backend: str = "ollama", model: str = None) -> bool:
    if backend == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    else:
        try:
            import ollama
            model = model or settings.ollama_model
            ollama.chat(model=model, messages=[{"role": "user", "content": "Hi"}], options={"num_predict": 5})
            return True
        except:
            return False
