from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time
from rag_baseline import answer, search

app = FastAPI(
    title="RAG Baseline API",
    description="A simple Retrieval-Augmented Generation API using TF-IDF and rule-based generation",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4

class SearchRequest(BaseModel):
    query: str
    top_k: int = 4

class AnswerResponse(BaseModel):
    question: str
    answer: str
    processing_time: float
    retrieved_docs: int

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    processing_time: float

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Baseline API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask - Ask a question and get an answer",
            "search": "/search - Search for relevant documents",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if the index is available
        test_results, _ = search("test", k=1)
        return {
            "status": "healthy",
            "message": "RAG system is operational",
            "indexed_documents": len(test_results) > 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"RAG system error: {str(e)}"
        }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer from the RAG system"""
    try:
        start_time = time.time()
        
        # Get answer from RAG system
        result = answer(request.question, k=request.top_k)
        
        # Also get the retrieved documents for metadata
        docs, _ = search(request.question, k=request.top_k)
        
        processing_time = time.time() - start_time
        
        return AnswerResponse(
            question=request.question,
            answer=result,
            processing_time=processing_time,
            retrieved_docs=len(docs)
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, 
            detail="RAG index not found. Please run the ingestion process first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for relevant documents without generating an answer"""
    try:
        start_time = time.time()
        
        docs, scores = search(request.query, k=request.top_k)
        
        processing_time = time.time() - start_time
        
        # Format results
        results = []
        for doc, score in zip(docs, scores):
            results.append({
                "path": doc["path"],
                "text_preview": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "similarity_score": score,
                "text_length": len(doc["text"])
            })
        
        return SearchResponse(
            query=request.query,
            results=results,
            processing_time=processing_time
        )
    
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="RAG index not found. Please run the ingestion process first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get statistics about the RAG system"""
    try:
        # Try to get some basic stats
        test_docs, _ = search("machine learning", k=100)  # Get many docs to count total
        
        return {
            "total_documents": len(test_docs),
            "system_type": "TF-IDF based RAG",
            "features": [
                "Document retrieval with TF-IDF",
                "Rule-based answer generation",
                "RESTful API interface",
                "Evaluation metrics"
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
