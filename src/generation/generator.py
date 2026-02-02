"""
LLM-based answer generation with citation support.
"""

import ollama
from dataclasses import dataclass

from src.config import settings
from src.retrieval.retriever import RetrievedChunk


@dataclass
class GeneratedAnswer:
    """An answer with its sources."""
    answer: str
    sources: list[RetrievedChunk]
    query: str
    model: str
    
    def format_with_citations(self) -> str:
        """Format answer with numbered citations."""
        if not self.sources:
            return self.answer
        
        citations = "\n\n**Sources:**\n"
        for i, source in enumerate(self.sources, 1):
            citations += f"{i}. {source.citation}\n"
        
        return self.answer + citations


SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in Chinese zodiac signs and numerology. 
Answer questions based ONLY on the provided context. If the context doesn't contain enough information, say so.

Guidelines:
- Be accurate and cite which zodiac sign or life path number you're discussing
- When discussing compatibility, mention both parties
- Keep answers concise but complete
- If asked about something not in the context, say "I don't have information about that."

Context will be provided in <context> tags."""


class Generator:
    """Generate answers using Ollama LLM with retrieved context."""
    
    def __init__(self, model: str = None):
        self.model = model or settings.ollama_model
    
    def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> GeneratedAnswer:
        """Generate an answer based on retrieved context."""
        context_text = self._format_context(context_chunks)
        
        user_prompt = f"""<context>
{context_text}
</context>

Question: {query}

Answer based on the context above."""
        
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        
        answer_text = response["message"]["content"]
        
        return GeneratedAnswer(
            answer=answer_text,
            sources=context_chunks,
            query=query,
            model=self.model,
        )
    
    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks into context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {chunk.citation}]\n{chunk.content}")
        return "\n\n---\n\n".join(parts)
    
    def generate_streaming(self, query: str, context_chunks: list[RetrievedChunk], temperature: float = 0.7):
        """Generate answer with streaming output."""
        context_text = self._format_context(context_chunks)
        
        user_prompt = f"""<context>
{context_text}
</context>

Question: {query}

Answer based on the context above."""
        
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
            stream=True,
        )
        
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]


def check_llm_available(model: str = None) -> bool:
    """Check if LLM is available."""
    model = model or settings.ollama_model
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            options={"num_predict": 5},
        )
        return "message" in response
    except Exception as e:
        print(f"LLM not available: {e}")
        print(f"Run: ollama pull {model}")
        return False
