#!/bin/bash

# Script to create and deploy the Hugging Face Space
# Run this after you've logged in with: huggingface-cli login

echo "🚀 Creating Hugging Face Space for RAG Demo..."

# Navigate to space directory
cd space

# Initialize git repo
git init
echo "✅ Initialized git repository"

# Create the space on Hugging Face (replace YOUR_USERNAME with your actual username)
read -p "Enter your Hugging Face username: " HF_USERNAME
SPACE_NAME="p1-rag-demo"

echo "📡 Creating space: $HF_USERNAME/$SPACE_NAME"

# Create the space using HF CLI
huggingface-cli repo create $SPACE_NAME --type space --space_sdk streamlit --private false

# Add remote
git remote add origin https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME

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

echo "🎉 Space created successfully!"
echo "🌐 Your Space will be available at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo "⏱️  It may take 2-3 minutes to build and deploy."

# Return to parent directory
cd ..

echo "✅ Deployment complete! Check your Space in a few minutes."
