"""Streamlit Web UI with streaming, feedback, and analytics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import time

st.set_page_config(page_title="Mystic RAG", page_icon="🔮", layout="wide")

from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available
from src.analytics import record_feedback, get_summary_stats

@st.cache_resource
def load_pipeline():
    return RAGPipeline(use_reranker=True)

def main():
    st.title("🔮 Mystic RAG")
    st.caption("Chinese Zodiac & Numerology Intelligence")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Sources", 1, 10, 5)
        use_memory = st.checkbox("Use conversation memory", True)
        
        st.divider()
        stats = get_summary_stats()
        st.header("📊 Stats")
        col1, col2 = st.columns(2)
        col1.metric("Queries", stats["total_queries"])
        col2.metric("Cost", f"${stats['total_cost_usd']:.4f}")
        thumbs_up = stats["thumbs_up"] or 0
        thumbs_down = stats["thumbs_down"] or 0
        st.caption(f"👍 {thumbs_up}  👎 {thumbs_down}")
        
        st.divider()
        st.header("💡 Examples")
        examples = ["What does it mean to be a Dragon?", "What careers suit that sign?", "Are Rat and Ox compatible?"]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:15]}"):
                st.session_state.query = ex
        
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_ids = {}
            st.cache_resource.clear()
            st.rerun()
    
    if not check_llm_available():
        st.error("LLM not available. Run: ollama serve")
        return
    
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_ids" not in st.session_state:
        st.session_state.query_ids = {}
    
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                meta = msg["metadata"]
                with st.expander(f"Sources ({meta.get('latency_ms', 0):.0f}ms)"):
                    if meta.get("rewritten_query"):
                        st.caption(f"🔄 Rewritten: *{meta['rewritten_query']}*")
                    for j, src in enumerate(meta.get("sources", [])):
                        score = meta.get("scores", [])[j] if j < len(meta.get("scores", [])) else 0
                        st.caption(f"**{src}** ({score:.2f})")
            if msg["role"] == "assistant" and i in st.session_state.query_ids:
                query_id = st.session_state.query_ids[i]
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"up_{i}"):
                        record_feedback(query_id, 1)
                        st.toast("Thanks!")
                with col2:
                    if st.button("👎", key=f"down_{i}"):
                        record_feedback(query_id, -1)
                        st.toast("Thanks!")
    
    query = st.session_state.pop("query", None)
    if prompt := st.chat_input("Ask about zodiac or numerology...") or query:
        user_input = prompt or query
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = pipeline.query(user_input, top_k=top_k, use_memory=use_memory, rewrite_query=True)
            st.write(response.answer)
            
            latency_ms = response.latency_ms
            cost_str = f", ${response.cost_usd:.6f}" if response.cost_usd > 0 else ""
            status = "reranked" if response.reranked else "vector"
            if response.rewritten_query:
                status += ", rewritten"
            
            with st.expander(f"Sources ({latency_ms:.0f}ms, {status}{cost_str})"):
                if response.rewritten_query:
                    st.caption(f"🔄 Rewritten: *{response.rewritten_query}*")
                for j, s in enumerate(response.sources):
                    st.caption(f"**{s.citation}** ({response.retrieval_scores[j]:.2f})")
            
            msg_idx = len(st.session_state.messages)
            st.session_state.query_ids[msg_idx] = response.query_id
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "metadata": {
                    "latency_ms": latency_ms,
                    "sources": [s.citation for s in response.sources],
                    "scores": response.retrieval_scores,
                    "rewritten_query": response.rewritten_query
                }
            })
            
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key="up_new"):
                    record_feedback(response.query_id, 1)
                    st.toast("Thanks!")
            with col2:
                if st.button("👎", key="down_new"):
                    record_feedback(response.query_id, -1)
                    st.toast("Thanks!")

if __name__ == "__main__":
    main()
