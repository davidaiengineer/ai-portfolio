#!/bin/bash

# AI Portfolio Setup Script
# This script sets up the development environment for the AI portfolio project

set -e  # Exit on any error

echo "🚀 Setting up AI Portfolio Development Environment"
echo "=================================================="

# Check if Python 3.11+ is installed
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is installed"
else
    echo "❌ Python 3.11+ is required. Please install Python 3.11 or higher."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed outputs logs models/cache models/hf_cache

# Set up Jupyter kernel
echo "🔬 Setting up Jupyter kernel..."
python -m ipykernel install --user --name ai-portfolio --display-name "AI Portfolio"

# Copy environment file
echo "⚙️ Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Environment file created from template"
    echo "📝 Please edit .env file with your API keys and configuration"
else
    echo "✅ Environment file already exists"
fi

# Install pre-commit hooks (optional)
echo "🔍 Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️ Pre-commit not found. Install with: pip install pre-commit"
fi

# Run initial tests
echo "🧪 Running initial tests..."
if command -v pytest &> /dev/null; then
    pytest -q || echo "⚠️ Some tests failed, but setup continues"
else
    echo "⚠️ pytest not found, skipping tests"
fi

echo ""
echo "🎉 Setup complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Edit .env file with your API keys"
echo "3. Run 'make help' to see available commands"
echo "4. Start with the basics notebook: jupyter notebook notebooks/001_basics.ipynb"
echo ""
echo "Happy coding! 🚀"
