"""
RAGAS Evaluation for Mystic RAG.
Measures: Faithfulness, Answer Relevance, Context Precision
"""

import json
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.rag_pipeline import RAGPipeline


@dataclass
class RAGASResult:
    query: str
    answer: str
    contexts: list[str]
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0


# Test cases with ground truth
TEST_CASES = [
    {
        "query": "What are the personality traits of a Rat?",
        "ground_truth": "Rats are clever, resourceful, quick-witted, adaptable"
    },
    {
        "query": "Are Dragon and Monkey compatible?",
        "ground_truth": "Dragon and Monkey are highly compatible, same triangle"
    },
    {
        "query": "What careers suit Life Path 7?",
        "ground_truth": "Research, philosophy, science, psychology, analysis"
    },
    {
        "query": "What is a Master Number in numerology?",
        "ground_truth": "11, 22, 33 are master numbers with amplified energy"
    },
    {
        "query": "What element years exist for Snake?",
        "ground_truth": "Wood Snake, Fire Snake, Earth Snake, Metal Snake, Water Snake"
    },
]


class RAGASEvaluator:
    """Simple RAGAS-style evaluation without external API dependencies."""
    
    def __init__(self):
        self.console = Console()
        self.pipeline = RAGPipeline(use_reranker=True)
    
    def evaluate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """
        Check if answer is grounded in context.
        Simple heuristic: % of answer sentences that overlap with context.
        """
        if not answer or not contexts:
            return 0.0
        
        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                     'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                     'who', 'whom', 'this', 'that', 'these', 'those', 'am'}
        
        answer_words = answer_words - stopwords
        context_words = context_words - stopwords
        
        if not answer_words:
            return 0.0
        
        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)
    
    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """
        Check if answer addresses the query.
        Simple heuristic: query term overlap with answer.
        """
        if not query or not answer:
            return 0.0
        
        query_words = set(query.lower().split()) - {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who', 'which', 'do', 'does', 'can', 'tell', 'me', 'about'}
        answer_lower = answer.lower()
        
        if not query_words:
            return 1.0
        
        matches = sum(1 for word in query_words if word in answer_lower)
        return matches / len(query_words)
    
    def evaluate_context_precision(self, query: str, contexts: list[str], ground_truth: str) -> float:
        """
        Check if retrieved contexts are relevant.
        Simple heuristic: ground truth terms in contexts.
        """
        if not contexts or not ground_truth:
            return 0.0
        
        context_text = " ".join(contexts).lower()
        truth_words = set(ground_truth.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'and', 'or', 'with'}
        
        if not truth_words:
            return 1.0
        
        matches = sum(1 for word in truth_words if word in context_text)
        return matches / len(truth_words)
    
    def run(self) -> list[RAGASResult]:
        """Run evaluation on test cases."""
        results = []
        
        self.console.print(Panel.fit(f"Running RAGAS evaluation on {len(TEST_CASES)} test cases", title="RAGAS"))
        
        for i, test in enumerate(TEST_CASES, 1):
            self.console.print(f"\n[{i}/{len(TEST_CASES)}] {test['query'][:50]}...")
            
            # Get RAG response
            response = self.pipeline.query(test['query'], use_memory=False)
            
            contexts = [c.content for c in response.sources]
            
            # Calculate metrics
            faithfulness = self.evaluate_faithfulness(response.answer, contexts)
            relevancy = self.evaluate_answer_relevancy(test['query'], response.answer)
            precision = self.evaluate_context_precision(test['query'], contexts, test['ground_truth'])
            
            result = RAGASResult(
                query=test['query'],
                answer=response.answer,
                contexts=contexts,
                faithfulness=faithfulness,
                answer_relevancy=relevancy,
                context_precision=precision,
            )
            results.append(result)
            
            self.console.print(f"  Faithfulness: {faithfulness:.2f} | Relevancy: {relevancy:.2f} | Precision: {precision:.2f}")
        
        return results
    
    def print_summary(self, results: list[RAGASResult]):
        """Print evaluation summary."""
        table = Table(title="RAGAS Evaluation Results")
        table.add_column("Query", width=35)
        table.add_column("Faithful", width=10)
        table.add_column("Relevant", width=10)
        table.add_column("Precision", width=10)
        
        for r in results:
            f_style = "green" if r.faithfulness > 0.6 else "yellow" if r.faithfulness > 0.3 else "red"
            r_style = "green" if r.answer_relevancy > 0.6 else "yellow" if r.answer_relevancy > 0.3 else "red"
            p_style = "green" if r.context_precision > 0.6 else "yellow" if r.context_precision > 0.3 else "red"
            
            table.add_row(
                r.query[:35],
                f"[{f_style}]{r.faithfulness:.2f}[/{f_style}]",
                f"[{r_style}]{r.answer_relevancy:.2f}[/{r_style}]",
                f"[{p_style}]{r.context_precision:.2f}[/{p_style}]",
            )
        
        self.console.print(table)
        
        # Averages
        avg_faith = sum(r.faithfulness for r in results) / len(results)
        avg_rel = sum(r.answer_relevancy for r in results) / len(results)
        avg_prec = sum(r.context_precision for r in results) / len(results)
        
        self.console.print(Panel.fit(
            f"Avg Faithfulness: {avg_faith:.2f}\n"
            f"Avg Answer Relevancy: {avg_rel:.2f}\n"
            f"Avg Context Precision: {avg_prec:.2f}\n"
            f"\nOverall Score: {(avg_faith + avg_rel + avg_prec) / 3:.2f}",
            title="Summary"
        ))


def main():
    evaluator = RAGASEvaluator()
    results = evaluator.run()
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
