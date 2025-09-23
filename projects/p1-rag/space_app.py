import time
import json
from pathlib import Path
import streamlit as st
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
from rag_baseline import search, answer

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "outputs" / "telemetry.jsonl"
(ROOT / "outputs").mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RAG Demo - AI Portfolio", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🧠 RAG Demo (TF-IDF + Rule-based)")
st.caption("**Deployed on Hugging Face Spaces** · Local TF-IDF retrieval + rule-based generation")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About This Demo")
    st.markdown("""
    **What:** Retrieval-Augmented Generation system
    
    **Tech Stack:**
    - 🔍 **Retrieval**: TF-IDF + Cosine Similarity
    - 🤖 **Generation**: Rule-based text processing  
    - 📚 **Corpus**: 5 AI/ML documents (15 chunks)
    - ⚡ **Speed**: Sub-second responses
    
    **Performance:**
    - F1 Score: 0.243
    - Coverage: 60% (6/10 questions)
    - Processing: ~0.01s per query
    """)
    
    st.header("🎯 Try These Questions")
    sample_questions = [
        "What is machine learning?",
        "What are CNNs used for?",
        "What is deep learning?",
        "What is NLP?",
        "What is computer vision?",
        "What are activation functions?",
        "What is overfitting?",
        "What is the data science process?"
    ]
    
    for q in sample_questions:
        if st.button(f"💡 {q}", key=f"btn_{hash(q)}", use_container_width=True):
            st.session_state.question = q

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "❓ **Ask a question about AI/ML:**", 
        value=st.session_state.get('question', ''),
        placeholder="e.g., What is machine learning?",
        key="question_input"
    )

with col2:
    top_k = st.slider("📊 Top K docs", 1, 8, 4, help="Number of documents to retrieve")

# Main action
if st.button("🚀 Ask", type="primary", use_container_width=True, key="ask_button_main") and question.strip():
    with st.spinner("🔍 Searching documents and generating answer..."):
        try:
            t0 = time.time()
            
            # Get documents and answer
            docs, scores = search(question, k=top_k)
            result = answer(question, k=top_k)
            
            latency = time.time() - t0
            
            # Display results
            st.success(f"✅ **Answer generated in {latency:.3f}s**")
            
            # Answer section
            st.subheader("💡 Answer")
            
            # Display answer with better visibility
            with st.container():
                st.markdown(f"""
                <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; border: 1px solid #d0e7ff;">
                    <p style="margin: 0; font-size: 16px; line-height: 1.6; color: #1a1a1a; font-weight: 500;">{result}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Backup display method with high contrast
                st.info(f"**Answer:** {result}")
            
            # Retrieved documents section
            st.subheader("📚 Retrieved Documents")
            
            for i, (doc, score) in enumerate(zip(docs, scores), 1):
                with st.expander(f"📄 **Document {i}** (Similarity: {score:.3f}) - {Path(doc['path']).name}"):
                    # Show document path
                    st.code(doc['path'], language="text")
                    
                    # Show preview
                    preview = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
                    st.markdown(f"**Preview:** {preview}")
                    
                    # Show full text in details
                    with st.expander("🔍 View Full Text"):
                        st.text(doc['text'])
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Latency", f"{latency:.3f}s")
            with col2:
                st.metric("📊 Retrieved", f"{len(docs)} docs")
            with col3:
                st.metric("🎯 Best Score", f"{max(scores):.3f}")
            with col4:
                st.metric("📝 Answer Length", f"{len(result)} chars")
            
            # Log telemetry
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": result,
                "top_k": top_k,
                "latency_s": round(latency, 3),
                "num_docs": len(docs),
                "best_score": round(max(scores), 3),
                "answer_length": len(result)
            }
            
            try:
                with LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                st.success("📝 Interaction logged to telemetry.jsonl")
            except Exception as e:
                st.warning(f"⚠️ Logging failed: {e}")
                
        except FileNotFoundError:
            st.error("❌ **RAG index not found!** Please run the ingestion process first.")
            st.code("python src/rag_baseline.py --ingest", language="bash")
            
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")

elif st.button("🚀 Ask", type="primary", use_container_width=True, key="ask_button_empty"):
    st.warning("⚠️ Please enter a question first!")

# Footer
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔗 Links:**
    - [GitHub Repository](https://github.com/davidaiengineer/ai-portfolio)
    - [Portfolio README](https://github.com/davidaiengineer/ai-portfolio#readme)
    """)

with col2:
    st.markdown("""
    **⚡ Quick Stats:**
    - Documents: 5 AI/ML guides
    - Chunks: 15 total
    - Vocabulary: 1,728 terms
    """)

with col3:
    st.markdown("""
    **🚀 Next Steps:**
    - Neural retrieval (sentence transformers)
    - LLM generation (FLAN-T5/GPT)
    - Re-ranking & conversation memory
    """)

# Instructions for first-time users
if not LOG.exists() or LOG.stat().st_size == 0:
    st.info("""
    👋 **Welcome!** This is a demonstration RAG system. Try asking questions about:
    - Machine Learning fundamentals
    - Deep Learning concepts  
    - Natural Language Processing
    - Computer Vision basics
    - Data Science workflows
    
    Click the sample questions in the sidebar to get started!
    """)

# Show recent activity if telemetry exists
if LOG.exists() and LOG.stat().st_size > 0:
    with st.expander("📊 Recent Activity", expanded=False):
        try:
            lines = LOG.read_text(encoding="utf-8").strip().split("\n")
            recent_logs = [json.loads(line) for line in lines[-5:]]  # Last 5 interactions
            
            st.subheader("Last 5 Interactions")
            for log in reversed(recent_logs):  # Show most recent first
                st.markdown(f"""
                **{log.get('timestamp', 'Unknown')}** - *{log.get('latency_s', 0):.3f}s*  
                **Q:** {log.get('question', 'N/A')[:100]}...  
                **A:** {log.get('answer', 'N/A')[:150]}...
                """)
                st.markdown("---")
                
        except Exception as e:
            st.error(f"Error loading telemetry: {e}")
