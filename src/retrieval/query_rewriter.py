"""Query Rewriting for better retrieval."""

import re

class QueryRewriter:
    def __init__(self, llm_backend: str = "ollama"):
        self.llm_backend = llm_backend
        
    def needs_rewriting(self, query: str, memory: list) -> bool:
        query_lower = query.lower().strip()
        if len(query_lower.split()) <= 3:
            return True
        pronouns = ["it", "they", "them", "that", "this", "those", "these", "their", "its"]
        if any(f" {p} " in f" {query_lower} " or query_lower.startswith(f"{p} ") for p in pronouns):
            return True
        followup_phrases = ["tell me more", "more about", "what about", "how about", "and what", "also", "what else", "that sign", "that number", "elaborate"]
        if any(phrase in query_lower for phrase in followup_phrases):
            return True
        return False
    
    def rewrite(self, query: str, memory: list) -> str:
        if not memory or not self.needs_rewriting(query, memory):
            return query
        return self._rule_based_rewrite(query, memory)
    
    def _rule_based_rewrite(self, query: str, memory: list) -> str:
        if not memory:
            return query
        last_topic = self._extract_topic(memory)
        if not last_topic:
            return query
        query_lower = query.lower()
        replacements = [
            (r"\bthat sign\b", last_topic),
            (r"\bthis sign\b", last_topic),
            (r"\bthe sign\b", last_topic),
            (r"\bthat number\b", last_topic),
            (r"\bit\b", last_topic),
            (r"\bthem\b", last_topic),
        ]
        rewritten = query
        for pattern, replacement in replacements:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        if any(phrase in query_lower for phrase in ["tell me more", "more about", "what else"]):
            rewritten = f"Tell me more about {last_topic}. {query}"
        if "career" in query_lower and ("that" in query_lower or "this" in query_lower):
            rewritten = f"What careers suit {last_topic}?"
        return rewritten
    
    def _extract_topic(self, memory: list) -> str:
        zodiac_signs = ["rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "goat", "sheep", "monkey", "rooster", "dog", "pig"]
        for msg in reversed(memory[-4:]):
            content_lower = msg.content.lower()
            for sign in zodiac_signs:
                if sign in content_lower:
                    return sign.capitalize()
        return ""

_rewriter = None

def get_rewriter(llm_backend: str = "ollama") -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter(llm_backend)
    return _rewriter
