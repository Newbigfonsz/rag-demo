"""LLM generation with memory support."""

import ollama
from dataclasses import dataclass
from src.config import settings
from src.retrieval.retriever import RetrievedChunk


@dataclass
class GeneratedAnswer:
    answer: str
    sources: list[RetrievedChunk]
    query: str
    model: str


SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in Chinese zodiac signs and numerology.

RULES:
1. ONLY use information from the provided context
2. If the context doesn't have the answer, say "I don't have information about that"
3. Be specific about which zodiac sign or life path number you're discussing
4. If there's conversation history, use it to understand follow-up questions like "that sign" or "what about careers"
5. Keep answers conversational but informative"""


class Generator:
    def __init__(self, model: str = None):
        self.model = model or settings.ollama_model
    
    def generate(self, query: str, context_chunks: list[RetrievedChunk], max_tokens: int = 1024, temperature: float = 0.7, memory_context: str = "") -> GeneratedAnswer:
        context = self._format_context(context_chunks)
        
        if memory_context:
            prompt = f"{memory_context}\n\n<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer based on context and conversation history."
        else:
            prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer based on the context."
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            options={"num_predict": max_tokens, "temperature": temperature},
        )
        
        return GeneratedAnswer(answer=response["message"]["content"], sources=context_chunks, query=query, model=self.model)
    
    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        return "\n\n---\n\n".join([f"[Source {i}: {c.citation}]\n{c.content}" for i, c in enumerate(chunks, 1)])
    
    def generate_streaming(self, query: str, context_chunks: list[RetrievedChunk], temperature: float = 0.7):
        context = self._format_context(context_chunks)
        prompt = f"<context>\n{context}\n</context>\n\nQuestion: {query}\n\nAnswer:"
        stream = ollama.chat(model=self.model, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], options={"temperature": temperature}, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]


def check_llm_available(model: str = None) -> bool:
    model = model or settings.ollama_model
    try:
        return "message" in ollama.chat(model=model, messages=[{"role": "user", "content": "Hi"}], options={"num_predict": 5})
    except:
        return False
