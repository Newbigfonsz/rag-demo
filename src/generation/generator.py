"""Generator with automatic fallback."""

import os
from dataclasses import dataclass
import ollama
from anthropic import Anthropic

CLAUDE_INPUT_COST = 3.00 / 1_000_000
CLAUDE_OUTPUT_COST = 15.00 / 1_000_000

def calculate_claude_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * CLAUDE_INPUT_COST) + (output_tokens * CLAUDE_OUTPUT_COST)

@dataclass
class GeneratedAnswer:
    answer: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

def check_llm_available(backend: str = "ollama") -> bool:
    if backend == "ollama":
        try:
            ollama.list()
            return True
        except:
            return False
    elif backend == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False

class Generator:
    def __init__(self, backend: str = "ollama", auto_fallback: bool = True):
        self.backend = backend
        self.auto_fallback = auto_fallback
        self.model = "llama3.2" if backend == "ollama" else "claude-sonnet-4-20250514"
        if backend == "claude" or auto_fallback:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self.claude_client = Anthropic(api_key=api_key) if api_key else None
    
    def generate(self, query: str, chunks: list, memory_context: str = "") -> GeneratedAnswer:
        context = "\n\n".join([f"[{c.citation}]\n{c.text}" for c in chunks])
        system_prompt = """You are a helpful assistant specializing in Chinese zodiac and numerology.
Answer based ONLY on the provided context. If the context doesn't contain the answer, say so.
Be concise but informative."""
        
        user_prompt = f"{memory_context}\n\nContext:\n{context}\n\nQuestion: {query}" if memory_context else f"Context:\n{context}\n\nQuestion: {query}"
        
        # Try primary backend
        if self.backend == "ollama":
            try:
                return self._generate_ollama(system_prompt, user_prompt)
            except Exception as e:
                if self.auto_fallback and self.claude_client:
                    print(f"[FALLBACK] Ollama failed ({e}), using Claude")
                    return self._generate_claude(system_prompt, user_prompt)
                raise
        else:
            return self._generate_claude(system_prompt, user_prompt)
    
    def _generate_ollama(self, system_prompt: str, user_prompt: str, timeout: int = 60) -> GeneratedAnswer:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"num_predict": 500}
        )
        return GeneratedAnswer(
            answer=response["message"]["content"],
            model=self.model,
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            cost_usd=0.0
        )
    
    def _generate_claude(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
        if not self.claude_client:
            raise ValueError("Claude API key not set")
        response = self.claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return GeneratedAnswer(
            answer=response.content[0].text,
            model="claude-sonnet-4-20250514",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_claude_cost(input_tokens, output_tokens)
        )
