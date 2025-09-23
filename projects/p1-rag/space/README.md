---
title: RAG Demo - AI Portfolio
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.49.1
app_file: app.py
pinned: false
license: mit
---

# 🧠 RAG Demo - AI Portfolio

**What:** A Retrieval-Augmented Generation (RAG) system demonstrating document retrieval and answer generation for AI/ML questions.

**How:** Type a question about AI/ML topics; the app retrieves relevant document chunks and generates grounded answers.

## 🏗️ Architecture

This RAG baseline system uses:
- **🔍 Document Retrieval**: TF-IDF vectorization with cosine similarity
- **🤖 Answer Generation**: Rule-based text processing with keyword matching
- **📚 Knowledge Base**: 5 comprehensive AI/ML documents (15 chunks total)
- **⚡ Performance**: Sub-second response times (~0.01s per query)

## 📊 Performance Metrics

- **Average F1 Score**: 0.243
- **Answer Coverage**: 60% (6/10 test questions)
- **Processing Speed**: ~0.01s per query
- **Vocabulary Size**: 1,728 terms

## 🎯 Try These Topics

Ask questions about:
- **Machine Learning**: supervised/unsupervised learning, algorithms, evaluation
- **Deep Learning**: neural networks, CNNs, activation functions, training
- **NLP**: transformers, BERT, GPT, text processing, embeddings
- **Computer Vision**: image processing, object detection, CNNs
- **Data Science**: workflows, feature engineering, model deployment

## 🔧 Technical Details

### Retrieval System
- **Method**: TF-IDF with scikit-learn
- **Features**: 10K max features, 1-2 grams, English stop words removed
- **Similarity**: Cosine similarity for document ranking
- **Chunking**: 600 words with 120-word overlap

### Answer Generation
- **Approach**: Rule-based keyword matching and sentence extraction
- **Fallback**: Returns "couldn't find answer" for low-confidence queries
- **Context**: Uses top-K retrieved document chunks
- **Output**: Truncated to 500 characters for readability

### Knowledge Base
The system includes comprehensive guides on:
1. **Machine Learning Basics** - Types, algorithms, applications
2. **Deep Learning Overview** - Neural networks, architectures, training
3. **Natural Language Processing** - NLP tasks, models, applications
4. **Computer Vision Fundamentals** - Image processing, CNNs, applications
5. **Data Science Workflow** - End-to-end process, tools, best practices

## 🚀 Future Enhancements

This baseline system is designed for easy upgrades:
- **Neural Retrieval**: Upgrade to sentence transformers (all-MiniLM-L6-v2)
- **LLM Generation**: Add FLAN-T5 or GPT for better answer quality
- **Re-ranking**: Cross-encoder models for improved relevance
- **Conversation Memory**: Multi-turn dialogue capabilities

## 📈 Evaluation

Tested on 10 curated questions covering:
- Machine Learning fundamentals
- Deep Learning concepts
- NLP and Computer Vision basics
- Data Science processes

**Sample Results:**
- "What is machine learning?" → F1: 0.595 ✅
- "What are activation functions?" → F1: 0.360 ✅
- "What is TF-IDF?" → F1: 0.270 ✅

## 🔗 Links

- **GitHub Repository**: [davidaiengineer/ai-portfolio](https://github.com/davidaiengineer/ai-portfolio)
- **Project Documentation**: [P1-RAG README](https://github.com/davidaiengineer/ai-portfolio/tree/main/projects/p1-rag)
- **Portfolio**: [AI Engineer Portfolio](https://github.com/davidaiengineer/ai-portfolio#readme)

## ⚡ Quick Start

1. Enter your question in the text input
2. Adjust "Top K docs" slider if needed (default: 4)
3. Click "🚀 Ask" to get your answer
4. Explore retrieved documents and similarity scores
5. Check the sidebar for sample questions

## 📝 Data Sources & Credits

- **Knowledge Base**: Original AI/ML content created for this demo
- **Framework**: Built with Streamlit, scikit-learn, and FastAPI
- **Deployment**: Hosted on Hugging Face Spaces
- **License**: MIT License

---

*This is a demonstration project showcasing RAG fundamentals. For production use, consider upgrading to neural retrieval and LLM-based generation.*
