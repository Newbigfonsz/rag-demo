"""Generator with retry logic, auto-fallback, and alerting."""

import os
import time
from dataclasses import dataclass
from functools import wraps
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
    used_fallback: bool = False
    retries: int = 0

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs), attempt
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"[RETRY] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

class AlertManager:
    """Simple alerting system - logs alerts and can be extended."""
    
    def __init__(self):
        self.alerts = []
    
    def send_alert(self, level: str, message: str, details: dict = None):
        """Log alert. Extend this to send to Slack/email/PagerDuty."""
        alert = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message,
            "details": details or {}
        }
        self.alerts.append(alert)
        
        # Log to console
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "critical": "🔥"}.get(level, "📢")
        print(f"{emoji} [{level.upper()}] {message}")
        
        # TODO: Add integrations here
        # self._send_slack(alert)
        # self._send_email(alert)
    
    def get_recent_alerts(self, limit: int = 10) -> list:
        return self.alerts[-limit:]

# Global alert manager
alert_manager = AlertManager()

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
    def __init__(self, backend: str = "ollama", auto_fallback: bool = True, max_retries: int = 3):
        self.backend = backend
        self.auto_fallback = auto_fallback
        self.max_retries = max_retries
        self.model = "llama3.2" if backend == "ollama" else "claude-sonnet-4-20250514"
        self.fallback_count = 0
        
        if backend == "claude" or auto_fallback:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self.claude_client = Anthropic(api_key=api_key) if api_key else None
    
    def generate(self, query: str, chunks: list, memory_context: str = "") -> GeneratedAnswer:
        context = "\n\n".join([f"[{c.citation}]\n{c.text}" for c in chunks])
        system_prompt = """You are a helpful assistant specializing in Chinese zodiac and numerology.
Answer based ONLY on the provided context. If the context doesn't contain the answer, say so.
Be concise but informative."""
        
        user_prompt = f"{memory_context}\n\nContext:\n{context}\n\nQuestion: {query}" if memory_context else f"Context:\n{context}\n\nQuestion: {query}"
        
        used_fallback = False
        retries = 0
        
        if self.backend == "ollama":
            try:
                result, retries = self._generate_ollama_with_retry(system_prompt, user_prompt)
                if retries > 0:
                    alert_manager.send_alert("warning", f"Ollama succeeded after {retries} retries", {"query": query[:50]})
                return GeneratedAnswer(
                    answer=result.answer,
                    model=result.model,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost_usd=result.cost_usd,
                    used_fallback=False,
                    retries=retries
                )
            except Exception as e:
                if self.auto_fallback and self.claude_client:
                    self.fallback_count += 1
                    alert_manager.send_alert(
                        "error", 
                        f"Ollama failed after {self.max_retries} retries, falling back to Claude",
                        {"error": str(e), "fallback_count": self.fallback_count}
                    )
                    result = self._generate_claude(system_prompt, user_prompt)
                    return GeneratedAnswer(
                        answer=result.answer,
                        model=result.model,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cost_usd=result.cost_usd,
                        used_fallback=True,
                        retries=self.max_retries
                    )
                raise
        else:
            result = self._generate_claude(system_prompt, user_prompt)
            return result
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _generate_ollama_with_retry(self, system_prompt: str, user_prompt: str) -> GeneratedAnswer:
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
