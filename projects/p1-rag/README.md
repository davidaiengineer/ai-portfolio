# Project 1: RAG Baseline ✅

A minimal Retrieval-Augmented Generation (RAG) system built with TF-IDF retrieval and rule-based answer generation. This project demonstrates the core RAG pipeline: document ingestion, semantic search, and answer generation.

## 🎯 Project Overview

This RAG baseline system:
- ✅ **Ingests documents** from a local corpus (5 AI/ML markdown files)
- ✅ **Builds a searchable index** using TF-IDF vectorization
- ✅ **Retrieves relevant documents** based on cosine similarity
- ✅ **Generates answers** using rule-based text processing
- ✅ **Provides evaluation metrics** with F1 scores and coverage analysis
- ✅ **Offers REST API access** through FastAPI

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   TF-IDF Index   │    │   Answer Gen    │
│   (.md files)   │───▶│  (scikit-learn)  │───▶│  (Rule-based)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   FastAPI REST   │
                       │      Server      │
                       └──────────────────┘
```

## 📁 Project Structure

```
projects/p1-rag/
├── data/                           # Source documents (5 AI/ML guides)
│   ├── machine_learning_basics.md
│   ├── deep_learning_overview.md
│   ├── natural_language_processing.md
│   ├── computer_vision_fundamentals.md
│   └── data_science_workflow.md
├── src/                           # Source code
│   ├── rag_baseline.py           # Main RAG system
│   ├── eval.py                   # Evaluation framework
│   └── app.py                    # FastAPI REST API
├── index/                        # Generated search index
│   ├── tfidf_index.pkl          # TF-IDF vectors
│   ├── vectorizer.pkl           # Trained vectorizer
│   └── meta.jsonl               # Document metadata
├── outputs/                      # Results and evaluation
│   ├── eval.jsonl               # Evaluation questions/answers
│   └── eval_results.json        # Performance metrics
└── .venv/                       # Python virtual environment
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd projects/p1-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy scikit-learn pydantic fastapi uvicorn
```

### 2. Build Index
```bash
python src/rag_baseline.py --ingest
```

### 3. Ask Questions
```bash
python src/rag_baseline.py --ask "What is machine learning?"
python src/rag_baseline.py --ask "What are CNNs used for?" --topk 3
```

### 4. Run Evaluation
```bash
python src/eval.py
```

### 5. Start API Server
```bash
uvicorn src.app:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive API docs
```

## 📊 Performance Results

**Evaluation Summary** (10 test questions):
- **Average F1 Score**: 0.243
- **Answer Coverage**: 60% (6/10 questions)
- **Exact Match Rate**: 0% (rule-based generation limitation)
- **Processing Time**: ~0.01s per query

**Top Performing Questions**:
1. "What is machine learning?" - F1: 0.595 ✅
2. "What are activation functions?" - F1: 0.360 ✅
3. "What is TF-IDF?" - F1: 0.270 ✅

## 🔧 Technical Implementation

### Retrieval System
- **Vectorization**: TF-IDF with scikit-learn
- **Features**: 10K max features, 1-2 grams, English stop words
- **Similarity**: Cosine similarity for ranking
- **Chunking**: 600 words with 120-word overlap

### Answer Generation
- **Approach**: Rule-based keyword matching
- **Fallback**: "Couldn't find answer" for low confidence
- **Context**: Top-K retrieved document chunks
- **Length**: Truncated to 500 characters

### Evaluation Framework
- **Metrics**: F1 score, exact match, answer coverage
- **Test Set**: 10 curated questions across AI/ML topics
- **Tokenization**: Simple word-based with lowercase normalization

## 🌐 API Endpoints

- `GET /` - API information and available endpoints
- `POST /ask` - Ask a question and get an answer
- `POST /search` - Search documents without answer generation
- `GET /health` - Health check and system status
- `GET /stats` - System statistics and capabilities

**Example API Usage**:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is deep learning?", "top_k": 4}'
```

## 🎓 Key Learnings

### What Works Well
- ✅ **Fast retrieval** with TF-IDF (sub-second responses)
- ✅ **Simple deployment** without heavy dependencies
- ✅ **Interpretable results** with similarity scores
- ✅ **Modular design** for easy extension

### Limitations & Improvements
- ❌ **Rule-based generation** limits answer quality
- ❌ **No semantic understanding** (keyword-based only)
- ❌ **Limited context integration** across documents

### Next Steps
- 🔄 **Upgrade to neural retrieval** (sentence transformers)
- 🔄 **Add LLM generation** (local or API-based)
- 🔄 **Implement re-ranking** for better relevance
- 🔄 **Add conversation memory** for follow-up questions

## 📈 Evaluation Details

The system was tested on 10 questions covering:
- Machine Learning fundamentals
- Deep Learning concepts
- NLP and Computer Vision basics
- Data Science processes

**Sample Results**:
```
Q: What is machine learning?
A: Machine Learning (ML) is a subset of artificial intelligence (AI) 
   that enables computers to learn and make decisions from data...
F1: 0.595 | Coverage: ✅

Q: What are CNNs used for?
A: [Retrieved context about computer vision and image processing...]
F1: 0.163 | Coverage: ✅
```

## 🏆 Project Achievements

- ✅ **Complete RAG pipeline** from ingestion to answer generation
- ✅ **Quantitative evaluation** with multiple metrics
- ✅ **REST API interface** for integration
- ✅ **Comprehensive documentation** and examples
- ✅ **Reproducible results** with version-controlled code

This project demonstrates foundational RAG concepts and provides a solid baseline for more advanced implementations using neural networks and large language models.

## 🌐 Live Demo

- **🚀 Hugging Face Space**: [DavidAIEngineer/p1-rag-demo](https://huggingface.co/spaces/DavidAIEngineer/p1-rag-demo)
- **📱 Interactive Interface**: Full Streamlit UI with document exploration
- **📊 Real-time Metrics**: Processing speed, similarity scores, and telemetry
- **🎯 Sample Questions**: Pre-loaded questions covering all AI/ML topics

**Evidence**: `docs/evidence/p1-day6-space.png`

## 🚀 Demo Commands

Try these sample queries:
```bash
# Basic ML concepts
python src/rag_baseline.py --ask "What is supervised learning?"

# Deep learning
python src/rag_baseline.py --ask "What are neural networks?"

# Computer vision
python src/rag_baseline.py --ask "What is object detection?"

# Data science
python src/rag_baseline.py --ask "What is the data science workflow?"
```