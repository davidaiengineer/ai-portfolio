# 🚀 Manual Deployment Instructions

Since the CLI method had issues, here's how to deploy manually:

## Step 1: Create Space on Web Interface

1. Go to: https://huggingface.co/new-space
2. **Owner**: DavidAIEngineer
3. **Space name**: `p1-rag-demo`
4. **License**: MIT
5. **SDK**: Streamlit
6. **Hardware**: CPU basic (free)
7. **Visibility**: Public
8. Click **Create Space**

## Step 2: Push Files to Space

After creating the Space on the web:

```bash
# You're already in the space directory
cd /Users/mac/Downloads/ai-portfolio/projects/p1-rag/space

# Add the remote (replace with your actual space URL)
git remote add origin https://huggingface.co/spaces/DavidAIEngineer/p1-rag-demo

# Add all files
git add .

# Commit
git commit -m "🚀 Initial deployment: RAG Demo with TF-IDF retrieval

✅ Features:
- Streamlit UI with sidebar navigation
- TF-IDF document retrieval (15 chunks)
- Rule-based answer generation
- Telemetry logging
- Interactive document exploration
- Sample questions and metrics

📊 Performance:
- F1 Score: 0.243
- Coverage: 60%
- Speed: ~0.01s per query

🔗 Portfolio: https://github.com/davidaiengineer/ai-portfolio"

# Push to Hugging Face
git push -u origin main
```

## Step 3: Verify Deployment

- Your Space will be at: https://huggingface.co/spaces/DavidAIEngineer/p1-rag-demo
- Build time: 2-3 minutes
- Check logs if there are any issues

## Step 4: Update Portfolio Links

Once live, update the portfolio README with the correct URL.
