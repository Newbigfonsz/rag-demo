"""Streamlit Web UI for Mystic RAG."""

import streamlit as st
import time

st.set_page_config(page_title="Mystic RAG", page_icon="🔮", layout="wide")

from src.rag_pipeline import RAGPipeline
from src.generation.generator import check_llm_available

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

def main():
    st.title("🔮 Mystic RAG")
    st.caption("Chinese Zodiac & Numerology Intelligence")
    
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Sources", 1, 10, 5)
        
        st.divider()
        st.header("Examples")
        examples = [
            "What does it mean to be a Dragon?",
            "Are Rat and Ox compatible?",
            "What careers suit Life Path 7?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.query = ex
        
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    if not check_llm_available():
        st.error("LLM not available. Run: ollama pull llama3.2")
        return
    
    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    query = st.session_state.pop("query", None)
    
    if prompt := st.chat_input("Ask about zodiac or numerology...") or query:
        user_input = prompt or query
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                response = pipeline.query(user_input, top_k=top_k)
                latency = (time.time() - start) * 1000
            
            st.write(response.answer)
            
            with st.expander(f"Sources ({latency:.0f}ms)"):
                for i, s in enumerate(response.sources):
                    st.caption(f"**{s.citation}** ({response.retrieval_scores[i]:.2f})")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
            })

if __name__ == "__main__":
    main()
