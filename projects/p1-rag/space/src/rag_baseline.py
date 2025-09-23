import os
import json
import glob
import re
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Configuration
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    """Simple word-based chunker with overlap"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+size]
        if chunk_words:  # Only add non-empty chunks
            chunks.append(" ".join(chunk_words))
        i += size - overlap
    return [c for c in chunks if c.strip()]

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    # Remove markdown headers and formatting
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text.strip()

def load_corpus() -> List[Tuple[str, str]]:
    """Load and chunk all documents"""
    items = []
    print(f"Loading documents from {DATA_DIR}")
    
    for fp in glob.glob(str(DATA_DIR / "**/*.*"), recursive=True):
        if fp.endswith((".txt", ".md", ".mdx")):
            try:
                print(f"Processing: {Path(fp).name}")
                txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
                cleaned_txt = clean_text(txt)
                chunks = chunk(cleaned_txt)
                
                for i, c in enumerate(chunks):
                    items.append((f"{fp}:chunk_{i}", c))
                    
                print(f"  → {len(chunks)} chunks created")
            except Exception as e:
                print(f"Error processing {fp}: {e}")
    
    print(f"Total items: {len(items)}")
    return items

def build_index(items: List[Tuple[str, str]]):
    """Build TF-IDF index for retrieval"""
    print("Building TF-IDF index...")
    
    # Extract texts
    texts = [text for _, text in items]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit and transform texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Save index components
    index_path = INDEX_DIR / "tfidf_index.pkl"
    vectorizer_path = INDEX_DIR / "vectorizer.pkl"
    meta_path = INDEX_DIR / "meta.jsonl"
    
    # Save TF-IDF matrix
    with open(index_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    # Save vectorizer
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save metadata
    with open(meta_path, 'w', encoding='utf-8') as f:
        for path, text in items:
            f.write(json.dumps({"path": path, "text": text}) + "\n")
    
    print(f"Index built with {len(texts)} documents")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Saved to {INDEX_DIR}")

def search(query: str, k: int = 4) -> Tuple[List[dict], List[float]]:
    """Search for similar documents using TF-IDF cosine similarity"""
    # Load index components
    index_path = INDEX_DIR / "tfidf_index.pkl"
    vectorizer_path = INDEX_DIR / "vectorizer.pkl"
    meta_path = INDEX_DIR / "meta.jsonl"
    
    if not all(p.exists() for p in [index_path, vectorizer_path, meta_path]):
        raise FileNotFoundError("Index not found. Run with --ingest first.")
    
    # Load components
    with open(index_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load metadata
    with open(meta_path, 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f]
    
    # Vectorize query
    query_vector = vectorizer.transform([query])
    
    # Compute similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:k]
    top_scores = similarities[top_indices]
    
    # Return top documents with scores
    results = []
    scores = []
    for idx, score in zip(top_indices, top_scores):
        results.append(docs[idx])
        scores.append(float(score))
    
    return results, scores

def generate_simple_answer(question: str, context_docs: List[dict]) -> str:
    """Simple rule-based answer generation (placeholder for LLM)"""
    # This is a simplified version without LLM
    # In practice, you'd use a language model here
    
    context = "\n\n".join([d["text"][:1000] for d in context_docs])
    
    # Simple keyword-based response generation
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Find sentences that might contain the answer
    sentences = re.split(r'[.!?]+', context)
    relevant_sentences = []
    
    # Look for sentences containing question keywords
    question_words = re.findall(r'\w+', question_lower)
    question_words = [w for w in question_words if len(w) > 3]  # Filter short words
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Skip very short sentences
            sentence_lower = sentence.lower()
            # Count matching keywords
            matches = sum(1 for word in question_words if word in sentence_lower)
            if matches >= 1:  # At least one keyword match
                relevant_sentences.append((sentence, matches))
    
    if relevant_sentences:
        # Sort by number of matches and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]
        
        answer = ". ".join(top_sentences)
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        return f"Based on the documents: {answer}"
    else:
        return "I couldn't find a specific answer to that question in the provided documents."

def answer(query: str, k: int = 4) -> str:
    """Complete RAG pipeline: retrieve and generate"""
    docs, scores = search(query, k=k)
    
    # Print retrieval results for debugging
    print(f"\nRetrieved {len(docs)} documents:")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"{i+1}. Score: {score:.3f} | {doc['path']}")
        print(f"   Preview: {doc['text'][:100]}...")
    
    return generate_simple_answer(query, docs)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple RAG Baseline System")
    parser.add_argument("--ingest", action="store_true", help="Build index from data/")
    parser.add_argument("--ask", type=str, help="Ask a question")
    parser.add_argument("--topk", type=int, default=4, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    if args.ingest:
        print("🔄 Building RAG index...")
        items = load_corpus()
        if items:
            build_index(items)
            print("✅ Index built successfully!")
        else:
            print("❌ No documents found to index")
    
    if args.ask:
        print(f"🤔 Question: {args.ask}")
        try:
            t0 = time.time()
            ans = answer(args.ask, k=args.topk)
            elapsed = time.time() - t0
            
            print(f"\n💡 Answer: {ans}")
            print(f"\n⏱️  Time taken: {elapsed:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if not args.ingest and not args.ask:
        print("Usage:")
        print("  python src/rag_baseline.py --ingest")
        print("  python src/rag_baseline.py --ask 'What is machine learning?'")

if __name__ == "__main__":
    main()
