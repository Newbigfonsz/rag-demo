"""
Analytics and metrics tracking for Mystic RAG.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "analytics.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT,
            model TEXT,
            llm_backend TEXT,
            retrieval_scores TEXT,
            sources TEXT,
            latency_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            reranked INTEGER,
            feedback INTEGER,
            feedback_comment TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            total_queries INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0,
            avg_latency_ms REAL DEFAULT 0,
            thumbs_up INTEGER DEFAULT 0,
            thumbs_down INTEGER DEFAULT 0
        )
    """)
    
    conn.commit()
    conn.close()


@dataclass
class QueryMetrics:
    query: str
    answer: str
    model: str
    llm_backend: str
    retrieval_scores: list
    sources: list
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    reranked: bool = False
    
    def log(self) -> int:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO query_logs 
            (timestamp, query, answer, model, llm_backend, retrieval_scores, 
             sources, latency_ms, input_tokens, output_tokens, cost_usd, reranked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            self.query,
            self.answer,
            self.model,
            self.llm_backend,
            json.dumps(self.retrieval_scores),
            json.dumps(self.sources),
            self.latency_ms,
            self.input_tokens,
            self.output_tokens,
            self.cost_usd,
            1 if self.reranked else 0
        ))
        
        query_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return query_id


def record_feedback(query_id: int, feedback: int, comment: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE query_logs SET feedback = ?, feedback_comment = ? WHERE id = ?",
                   (feedback, comment, query_id))
    conn.commit()
    conn.close()


def get_recent_queries(limit: int = 50) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_summary_stats() -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*) as total_queries,
            COALESCE(SUM(cost_usd), 0) as total_cost_usd,
            COALESCE(AVG(latency_ms), 0) as avg_latency_ms,
            SUM(CASE WHEN feedback = 1 THEN 1 ELSE 0 END) as thumbs_up,
            SUM(CASE WHEN feedback = -1 THEN 1 ELSE 0 END) as thumbs_down,
            SUM(CASE WHEN llm_backend = 'claude' THEN 1 ELSE 0 END) as claude_queries,
            SUM(CASE WHEN llm_backend = 'ollama' THEN 1 ELSE 0 END) as ollama_queries
        FROM query_logs
    """)
    row = dict(cursor.fetchone())
    conn.close()
    return row


CLAUDE_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
}


def calculate_claude_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = CLAUDE_PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


init_db()
