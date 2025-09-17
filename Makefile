# AI Portfolio Makefile
# Common commands for development, testing, and deployment

.PHONY: help setup format test lint clean run-api run-rag run-vision run-mlops docker-build docker-run

# Default target
help:
	@echo "AI Portfolio - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          Install dependencies"
	@echo "  setup-dev      Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  format         Format code with ruff and black"
	@echo "  lint           Run linting checks"
	@echo "  test           Run all tests"
	@echo "  test-cov       Run tests with coverage"
	@echo ""
	@echo "Development:"
	@echo "  run-api        Start MLOps API server"
	@echo "  run-rag        Start RAG application"
	@echo "  run-vision     Start vision model demo"
	@echo "  run-mlops      Start MLOps service"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-run     Run services with Docker Compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean temporary files and caches"
	@echo "  install-hooks  Install pre-commit hooks"

# Setup
setup:
	pip install -r requirements.txt

setup-dev: setup
	pip install pre-commit
	pre-commit install

# Code Quality
format:
	ruff check . --fix
	black .
	isort .

lint:
	ruff check .
	black --check .
	isort --check-only .

test:
	pytest -v

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term

# Development servers
run-api:
	cd projects/p3-mlops && uvicorn src.app:app --reload --port 8000

run-rag:
	cd projects/p1-rag && streamlit run src/ui.py --server.port 8501

run-vision:
	cd projects/p2-vision && streamlit run src/demo.py --server.port 8502

run-mlops:
	cd projects/p3-mlops && python -m uvicorn src.app:app --reload --port 8000

# Docker
docker-build:
	docker build -t ai-portfolio-rag -f projects/p1-rag/Dockerfile projects/p1-rag/
	docker build -t ai-portfolio-vision -f projects/p2-vision/Dockerfile projects/p2-vision/
	docker build -t ai-portfolio-mlops -f projects/p3-mlops/Dockerfile projects/p3-mlops/

docker-run:
	docker-compose up --build

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

install-hooks:
	pre-commit install

# Project-specific commands
setup-rag:
	cd projects/p1-rag && pip install -r requirements.txt

setup-vision:
	cd projects/p2-vision && pip install -r requirements.txt

setup-mlops:
	cd projects/p3-mlops && pip install -r requirements.txt

# Data preparation
prepare-data:
	@echo "Preparing data directories..."
	mkdir -p data/raw data/processed
	@echo "Data directories created"

# Notebook setup
setup-notebooks:
	python -m ipykernel install --user --name ai-portfolio --display-name "AI Portfolio"

# Quick start
quick-start: setup prepare-data setup-notebooks
	@echo "AI Portfolio setup complete!"
	@echo "Run 'make help' to see available commands"
