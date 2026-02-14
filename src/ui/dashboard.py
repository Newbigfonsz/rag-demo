"""
Analytics Dashboard for Mystic RAG.
Run: streamlit run src/ui/dashboard.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Mystic RAG Analytics", page_icon="📊", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analytics import get_summary_stats, get_recent_queries, record_feedback

def main():
    st.title("📊 Mystic RAG Analytics")
    st.caption("Query logs, feedback, and cost tracking")
    
    # Get data
    stats = get_summary_stats()
    recent = get_recent_queries(100)
    
    # Summary cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Queries", stats['total_queries'])
    
    with col2:
        st.metric("Total Cost", f"${stats['total_cost_usd']:.4f}")
    
    with col3:
        avg_latency = stats['avg_latency_ms']
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    with col4:
        thumbs_up = stats['thumbs_up'] or 0
        thumbs_down = stats['thumbs_down'] or 0
        total_feedback = thumbs_up + thumbs_down
        satisfaction = (thumbs_up / total_feedback * 100) if total_feedback > 0 else 0
        st.metric("👍 Satisfaction", f"{satisfaction:.0f}%", f"{thumbs_up}👍 {thumbs_down}👎")
    
    with col5:
        claude = stats['claude_queries'] or 0
        ollama = stats['ollama_queries'] or 0
        st.metric("LLM Split", f"Claude: {claude}", f"Ollama: {ollama}")
    
    st.divider()
    
    # Charts
    if recent:
        df = pd.DataFrame(recent)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Latency Over Time")
            if len(df) > 1:
                chart_data = df[['timestamp', 'latency_ms']].set_index('timestamp').sort_index()
                st.line_chart(chart_data)
            else:
                st.info("Not enough data yet")
        
        with col2:
            st.subheader("💰 Cost by LLM")
            cost_by_llm = df.groupby('llm_backend')['cost_usd'].sum()
            if not cost_by_llm.empty:
                st.bar_chart(cost_by_llm)
            else:
                st.info("No cost data yet")
    
    st.divider()
    
    # Recent queries with feedback
    st.subheader("📝 Recent Queries")
    
    if not recent:
        st.info("No queries yet. Start chatting to see analytics!")
        return
    
    for i, q in enumerate(recent[:20]):
        with st.expander(f"**{q['query'][:60]}...** ({q['llm_backend']}, {q['latency_ms']:.0f}ms, ${q['cost_usd']:.6f})"):
            st.write("**Answer:**")
            st.write(q['answer'][:500] + "..." if len(q['answer'] or '') > 500 else q['answer'])
            
            st.write(f"**Model:** {q['model']}")
            st.write(f"**Tokens:** {q['input_tokens']} in / {q['output_tokens']} out")
            st.write(f"**Time:** {q['timestamp']}")
            
            # Feedback buttons
            current_feedback = q['feedback']
            
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                if current_feedback == 1:
                    st.success("👍 Liked")
                elif st.button("👍", key=f"up_{q['id']}"):
                    record_feedback(q['id'], 1)
                    st.rerun()
            
            with col2:
                if current_feedback == -1:
                    st.error("👎 Disliked")
                elif st.button("👎", key=f"down_{q['id']}"):
                    record_feedback(q['id'], -1)
                    st.rerun()
    
    st.divider()
    
    # Cost breakdown
    st.subheader("💵 Cost Breakdown")
    
    if recent:
        df = pd.DataFrame(recent)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By LLM Backend:**")
            cost_summary = df.groupby('llm_backend').agg({
                'cost_usd': 'sum',
                'query': 'count'
            }).rename(columns={'query': 'queries'})
            cost_summary['avg_cost'] = cost_summary['cost_usd'] / cost_summary['queries']
            st.dataframe(cost_summary)
        
        with col2:
            st.write("**Projections:**")
            total_queries = len(df)
            total_cost = df['cost_usd'].sum()
            avg_cost = total_cost / total_queries if total_queries > 0 else 0
            
            st.write(f"- Avg cost per query: **${avg_cost:.6f}**")
            st.write(f"- Est. 1,000 queries: **${avg_cost * 1000:.2f}**")
            st.write(f"- Est. 10,000 queries: **${avg_cost * 10000:.2f}**")


if __name__ == "__main__":
    main()
