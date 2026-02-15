"""Agentic RAG - Multi-step reasoning."""

from dataclasses import dataclass
from typing import List
from enum import Enum
import re

class AgentAction(Enum):
    SEARCH = "search"
    CLARIFY = "clarify"
    MULTI_SEARCH = "multi_search"

@dataclass
class AgentDecision:
    action: AgentAction
    reasoning: str
    search_queries: List[str] = None
    clarifying_question: str = None

class RAGAgent:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def analyze_query(self, query: str, memory: list = None) -> AgentDecision:
        query_lower = query.lower().strip()
        if self._is_too_vague(query_lower):
            return AgentDecision(action=AgentAction.CLARIFY, reasoning="Query is too vague", clarifying_question=self._generate_clarification(query_lower))
        if self._needs_multi_search(query_lower):
            queries = self._decompose_query(query_lower)
            return AgentDecision(action=AgentAction.MULTI_SEARCH, reasoning="Query requires multiple topics", search_queries=queries)
        return AgentDecision(action=AgentAction.SEARCH, reasoning="Standard single search", search_queries=[query])
    
    def _is_too_vague(self, query: str) -> bool:
        vague = ["help", "hi", "hello", "hey", "?"]
        words = query.split()
        if len(words) == 1 and words[0] not in self._get_known_topics():
            return True
        if query.strip("?!. ") in vague:
            return True
        return False
    
    def _needs_multi_search(self, query: str) -> bool:
        indicators = ["compare", "difference between", "versus", " vs ", "both"]
        return any(ind in query for ind in indicators)
    
    def _decompose_query(self, query: str) -> List[str]:
        topics = self._extract_topics(query)
        if len(topics) >= 2:
            return [f"Tell me about {topic}" for topic in topics]
        return [query]
    
    def _extract_topics(self, query: str) -> List[str]:
        found = []
        zodiac = ["rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "goat", "monkey", "rooster", "dog", "pig"]
        for sign in zodiac:
            if sign in query:
                found.append(sign.capitalize())
        life_paths = re.findall(r"life path (\d+)", query)
        for lp in life_paths:
            found.append(f"Life Path {lp}")
        return found
    
    def _generate_clarification(self, query: str) -> str:
        if "compatible" in query:
            return "Which two zodiac signs would you like me to check compatibility for?"
        if "career" in query:
            return "Which zodiac sign or life path number would you like career advice for?"
        return "Could you be more specific? I can help with Chinese zodiac signs (Rat, Ox, Tiger, etc.) or life path numbers (1-9, 11, 22, 33)."
    
    def _get_known_topics(self) -> set:
        return {"rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "goat", "monkey", "rooster", "dog", "pig", "numerology", "zodiac", "compatibility"}
    
    def run(self, query: str, use_memory: bool = True) -> dict:
        decision = self.analyze_query(query, self.pipeline.memory if use_memory else None)
        if decision.action == AgentAction.CLARIFY:
            return {"response": decision.clarifying_question, "action_taken": "clarify", "reasoning": decision.reasoning, "is_clarification": True}
        if decision.action == AgentAction.MULTI_SEARCH:
            for sub_query in decision.search_queries[:3]:
                self.pipeline.query(sub_query, top_k=3, use_memory=False)
            final = self.pipeline.query(query, use_memory=use_memory)
            return {"response": final.answer, "action_taken": "multi_search", "reasoning": decision.reasoning, "sub_queries": decision.search_queries, "sources": [s.citation for s in final.sources], "is_clarification": False}
        response = self.pipeline.query(query, use_memory=use_memory)
        return {"response": response.answer, "action_taken": "search", "reasoning": decision.reasoning, "sources": [s.citation for s in response.sources], "rewritten_query": response.rewritten_query, "is_clarification": False}
