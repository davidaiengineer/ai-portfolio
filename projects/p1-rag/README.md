# P1 - RAG (Retrieval-Augmented Generation) App

## Overview
A domain-aware question-answering system that combines retrieval with generation to provide accurate, contextually relevant answers.

## Goals
- Build a RAG system with document retrieval and answer generation
- Implement evaluation metrics (context hit rate, groundedness)
- Create a simple UI for interaction
- Deploy as a demo (Hugging Face Spaces/Streamlit)

## Stack
- **Core**: Transformers, FAISS/Chroma for vector search
- **UI**: Streamlit or Gradio
- **Models**: Local LLMs or API-based (OpenAI, Azure OpenAI)
- **Evaluation**: Custom metrics for retrieval and generation quality

## Project Structure
```
p1-rag/
├── src/
│   ├── prepare_corpus.py    # Document processing and vector store creation
│   ├── app.py              # Main application entry point
│   ├── ui.py               # Streamlit/Gradio interface
│   ├── retrieval.py        # Document retrieval logic
│   ├── generation.py       # Answer generation logic
│   └── evaluation.py       # Metrics and evaluation
├── data/                   # Project-specific data
├── tests/                  # Unit and integration tests
└── README.md
```

## Getting Started

### 1. Setup Environment
```bash
cd projects/p1-rag
cp ../../.env.example .env  # Add API keys if using hosted models
```

### 2. Prepare Data
```bash
python src/prepare_corpus.py --input ../../data/raw --store ./vector_store
```

### 3. Run Application
```bash
# Streamlit UI
streamlit run src/ui.py

# Or Gradio UI
python src/app.py
```

## Evaluation Metrics
- **Context Hit Rate**: Percentage of queries with relevant retrieved documents
- **Groundedness**: How well answers are supported by retrieved context
- **Exact Match**: Exact string match with ground truth answers
- **F1 Score**: Token-level overlap with ground truth

## Deliverables
- [ ] Working RAG system with retrieval and generation
- [ ] Evaluation notebook with metrics
- [ ] Demo deployment (Hugging Face Spaces/Streamlit)
- [ ] Short video demonstration
- [ ] Model card with limitations and use cases

## Next Steps
1. Implement basic retrieval system
2. Add answer generation with context
3. Create evaluation framework
4. Build user interface
5. Deploy and document
