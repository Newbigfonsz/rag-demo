"""Streamlit Web UI with feedback buttons."""

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
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Sources", 1, 10, 5)
        use_memory = st.checkbox("Use conversation memory", True)
        
        st.divider()
        
        # Quick stats
        stats = get_summary_stats()
        st.header("📊 Stats")
        st.metric("Queries", stats['total_queries'])
        st.metric("Cost", f"${stats['total_cost_usd']:.4f}")
        thumbs_up = stats['thumbs_up'] or 0
        thumbs_down = stats['thumbs_down'] or 0
        st.metric("Feedback", f"👍 {thumbs_up}  👎 {thumbs_down}")
        
        if st.button("📊 Full Dashboard", use_container_width=True):
            st.switch_page("pages/dashboard.py")
        
        st.divider()
        st.header("Examples")
        for ex in ["What does it mean to be a Dragon?", "What careers suit that sign?", "Are Rat and Ox compatible?"]:
            if st.button(ex, use_container_width=True):
                st.session_state.query = ex
        
        st.divider()
        if st.button("Clear Chat & Memory", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_ids = {}
            st.cache_resource.clear()
            st.rerun()
    
    # Check LLM
    if not check_llm_available():
        st.error("LLM not available. Run: ollama serve")
        return
    
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return
    
    # Initialize state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_ids" not in st.session_state:
        st.session_state.query_ids = {}
    
    # Display messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Feedback buttons for assistant messages
            if msg["role"] == "assistant" and i in st.session_state.query_ids:
                query_id = st.session_state.query_ids[i]
                feedback_key = f"feedback_{query_id}"
                
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"up_{i}"):
                        record_feedback(query_id, 1)
                        st.toast("Thanks for the feedback! 👍")
                with col2:
                    if st.button("👎", key=f"down_{i}"):
                        record_feedback(query_id, -1)
                        st.toast("Thanks for the feedback! 👎")
    
    # Handle input
    query = st.session_state.pop("query", None)
    
    if prompt := st.chat_input("Ask about zodiac or numerology...") or query:
        user_input = prompt or query
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = pipeline.query(user_input, top_k=top_k, use_memory=use_memory)
            
            st.write(response.answer)
            
            # Show metrics
            cost_str = f", ${response.cost_usd:.6f}" if response.cost_usd > 0 else ""
            status = "reranked" if response.reranked else "vector"
            with st.expander(f"Sources ({response.latency_ms:.0f}ms, {status}{cost_str})"):
                for i, s in enumerate(response.sources):
                    st.caption(f"**{s.citation}** ({response.retrieval_scores[i]:.2f})")
            
            # Store query_id for feedback
            msg_idx = len(st.session_state.messages)
            st.session_state.query_ids[msg_idx] = response.query_id
            st.session_state.messages.append({"role": "assistant", "content": response.answer})
            
            # Feedback buttons
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_new"):
                    record_feedback(response.query_id, 1)
                    st.toast("Thanks! 👍")
            with col2:
                if st.button("👎", key=f"down_new"):
                    record_feedback(response.query_id, -1)
                    st.toast("Thanks! 👎")


if __name__ == "__main__":
    main()
