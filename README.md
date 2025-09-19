# AI Engineering Portfolio
Build and showcase production-minded AI projects while earning free, employer-recognized credentials.

![build](https://img.shields.io/badge/build-passing-informational)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![last_updated](https://img.shields.io/badge/last_update-2025-01-27-lightgrey)

---

## Contents
- [Overview](#overview)
- [Stack](#stack)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Credentials Tracker](#credentials-tracker)
- [Projects](#projects)
  - [P1 — RAG App](#p1--rag-app)
  - [P2 — Vision / Multimodal](#p2--vision--multimodal)
  - [P3 — MLOps Service](#p3--mlops-service)
- [Notebooks](#notebooks)
- [Data & Secrets](#data--secrets)
- [Evaluation](#evaluation)
- [Roadmap](#roadmap)
- [Evidence & Sharing](#evidence--sharing)
- [Contributing / Issues](#contributing--issues)
- [License](#license)

---

## Overview
This repository tracks my 10–12 week **AI Engineering path**:
- Ship **3 portfolio projects** (LLM RAG, Vision, MLOps).
- Earn **free credentials** (Google Cloud, Microsoft Applied Skills, Databricks, etc.).
- Publish demos (Hugging Face Spaces / Streamlit) + concise write-ups.

**Public repo = public accountability.** Milestones and badges are logged in `docs/evidence/`.

---

## Stack
- **Language**: Python 3.11+
- **Core**: PyTorch, Hugging Face (Transformers/Datasets), FAISS/Chroma, Gradio/Streamlit, FastAPI
- **Cloud/Labs (free tiers/badges)**: Google Cloud Skills Boost (Vertex AI), Microsoft Learn (Applied Skills), Databricks Academy
- **Dev**: Poetry or pip, Ruff, PyTest, Docker, GitHub Actions (optional)

---

## Folder Structure
```
ai-portfolio/
├─ projects/
│  ├─ p1-rag/           # LLM Retrieval-Augmented Generation app
│  ├─ p2-vision/        # Vision/multimodal model + demo
│  └─ p3-mlops/         # Train/serve API + CI/CD
├─ notebooks/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ scripts/
├─ docs/
│  ├─ notes/
│  ├─ evidence/
│  ├─ diagrams/
│  └─ assets/
├─ .github/workflows/
├─ .gitignore
├─ README.md
├─ requirements.txt or pyproject.toml
├─ Dockerfile
└─ Makefile
```

---

## Getting Started

### 1) Environment
Choose **one** setup path.

**Poetry**
```bash
pipx install poetry ruff pre-commit
poetry init -n
poetry add "torch>=2.2" transformers datasets faiss-cpu chromadb gradio streamlit fastapi uvicorn pydantic python-dotenv
poetry add -D pytest ruff ipykernel
pre-commit sample-config > .pre-commit-config.yaml
```

**pip**
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install "torch>=2.2" transformers datasets faiss-cpu chromadb gradio streamlit fastapi uvicorn pydantic python-dotenv pytest ruff ipykernel
```

### 2) Dev Quality (optional but recommended)
```bash
ruff check .
pytest -q || true
```

### 3) Run notebooks
```bash
python -m ipykernel install --user --name ai-portfolio
```

### 4) Makefile helpers (optional)
```makefile
# Makefile (drop in project root)
.PHONY: setup format test run-api
setup: ; pip install -r requirements.txt || poetry install
format: ; ruff check . --fix
test: ; pytest -q
run-api: ; uvicorn projects.p3-mlops.src.app:app --reload --port 8000
```

---

## Credentials Tracker

| Status | Program         | Credential                                 | Link  | Evidence                                   | Date   |
| ------ | --------------- | ------------------------------------------ | ----- | ------------------------------------------ | ------ |
| [x]   | Google Cloud    | Generative AI Fundamentals (Skill Badge)   | https://www.cloudskillsboost.google/paths/118 | docs/evidence/gcp-genai-fundamentals.png   | 2025-01-18 |
| [x]   | Google Cloud    | Prompt Design in Vertex AI (Badge)         | https://www.cloudskillsboost.google/course_templates/976 | docs/evidence/Prompt Design in Vertex AI.png | 2025-01-19 |
| [ ]   | Microsoft       | Applied Skills: Azure OpenAI Agents        | https://learn.microsoft.com/en-us/credentials/applied-skills/build-natural-language-solution-azure-openai/ | docs/evidence/ms-applied-skills-agents.png | {date} |
| [ ]   | Microsoft       | Applied Skills: Azure AI Vision            | https://learn.microsoft.com/en-us/credentials/applied-skills/create-computer-vision-solutions-azure-ai/ | docs/evidence/ms-applied-skills-vision.png | {date} |
| [ ]   | Databricks      | Generative AI Fundamentals (Accreditation) | https://www.databricks.com/learn/training/generative-ai-fundamentals | docs/evidence/dbx-genai.png                | {date} |
| [ ]   | IBM SkillsBuild | AI Fundamentals (Badge)                    | https://skillsbuild.org/adult-learners/course-catalog/artificial-intelligence | docs/evidence/ibm-ai-fundamentals.png      | {date} |
| [ ]   | Hugging Face    | Course Certificate (track)                 | https://huggingface.co/learn | docs/evidence/hf-learn.png                 | {date} |
| [x]   | Kaggle          | Intro to Machine Learning (Certificate)    | https://www.kaggle.com/learn/intro-to-machine-learning | docs/evidence/DavidAIEngineer - Intro to Machine Learning.png | 2025-01-17 |
| [x]   | Google Cloud    | Introduction to Generative AI (Completion) | https://www.cloudskillsboost.google/course_templates/536 | docs/evidence/Introduction to Generative AI.png | 2025-01-18 |
| [x]   | Google Cloud    | Introduction to Large Language Models (Completion) | https://www.cloudskillsboost.google/course_templates/539 | docs/evidence/Introduction to Large Language Models.png | 2025-01-18 |
| [x]   | Google Cloud    | Introduction to Responsible AI (Completion) | https://www.cloudskillsboost.google/course_templates/554 | docs/evidence/Introduction to Responsible AI.png | 2025-01-18 |

---

## Projects

### P1 — RAG App

* **Goal:** Domain-aware Q&A with retrieval, basic evaluation, and a simple UI.
* **Stack:** Transformers, FAISS/Chroma, FastAPI or Streamlit/Gradio.
* **Run (example):**
  ```bash
  cd projects/p1-rag
  cp .env.example .env   # add keys if needed; local models work without keys
  python src/prepare_corpus.py --input ../../data/raw --store ./vector_store
  python src/app.py      # or: streamlit run src/ui.py
  ```
* **Deliverables:**
  * Decomposition notebook (`notebooks/1xx_rag_baseline.ipynb`)
  * Metrics table (context hit rate, groundedness)
  * Demo link (HF Space/Streamlit) + short video

### P2 — Vision / Multimodal

* **Goal:** Train a small classifier or run a captioning/search demo; include a model card.
* **Stack:** PyTorch/timm, CLIP/BLIP (inference), optional Azure Vision labs.
* **Run (example):**
  ```bash
  cd projects/p2-vision
  python src/train.py --data ../../data/raw/your_dataset --epochs 5
  python src/infer.py --image docs/assets/sample.jpg
  ```
* **Deliverables:** Confusion matrix, augmentations summary, model card with limitations.

### P3 — MLOps Service

* **Goal:** Reproducible training + serving API with tests and CI.
* **Stack:** FastAPI, Docker, GitHub Actions.
* **Run (example):**
  ```bash
  cd projects/p3-mlops
  docker build -t mlops-service .
  docker run -p 8000:8000 mlops-service
  # open http://localhost:8000/docs
  ```
* **Deliverables:** CI badge, load test snapshot, deployment notes.

---

## Notebooks

* `notebooks/001_basics.ipynb` — Day-1 refresher (pandas + tiny model).
* `notebooks/1xx_rag_baseline.ipynb` — Chunking, embedding, retrieval eval.
* `notebooks/2xx_vision_baseline.ipynb` — Small CNN/ViT baseline.
* `notebooks/3xx_mlops_eval.ipynb` — E2E eval + latency profiling.

Keep notebooks deterministic and small; move heavy logic to `projects/*/src/`.

---

## Data & Secrets

* **Data:** put raw files in `data/raw/` and outputs in `data/processed/`. Avoid committing large files.
* **Secrets:** never commit keys. Use `.env` files locally.

`.env.example`
```ini
# Optional; needed only if you call hosted APIs
OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
GOOGLE_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
```

---

## Evaluation

* **LLM (P1):** Context-recall rate, groundedness, exact match / F1 on a small gold set.
* **Vision (P2):** Accuracy/F1, calibration (ECE), confusion matrix.
* **Service (P3):** p50/p95 latency, throughput, error rate.
* Store results in `projects/*/outputs/metrics.json` and summarize in each README.

---

## Roadmap

* **Weeks 1–2:** Kaggle certs + Google Cloud GenAI badges; RAG baseline.
* **Weeks 3–4:** Agents (Microsoft Applied Skills) + tool use; P1 v1 demo.
* **Weeks 5–6:** Guardrails & eval; start Vision; Azure Vision badge.
* **Weeks 7–8:** Databricks accreditation; HF Agents track; deploy to Spaces.
* **Weeks 9–10:** MLOps service + CI; choose capstone track.
* **Weeks 11–12:** Polish docs, publish case study, apply/interview.

---

## Evidence & Sharing

* Add screenshots/pdfs to `docs/evidence/`.
* Each week, log wins/blockers in `docs/notes/week-XX.md`.
* Post short updates; add links under a **Changelog** section in this README.

**Changelog**

* 2025-01-27 — Project structure created and initialized.
* 2025-01-17 — ✅ Completed Kaggle Intro to Machine Learning certificate.
* 2025-01-18 — ✅ Completed 3 Google Cloud AI courses: Intro to Generative AI, LLMs, and Responsible AI.
* 2025-01-18 — ✅ Earned Google Cloud Generative AI Fundamentals badge and built Prompt Playground app.
* 2025-01-19 — ✅ Completed Prompt Design in Vertex AI course and starting RAG baseline project.
* 2025-01-19 — ✅ Built complete RAG baseline with TF-IDF retrieval, evaluation framework, and REST API.

---

## Contributing / Issues

Issues and PRs are welcome for small fixes or ideas. File an issue with:

* What you tried
* Expected vs. actual
* Minimal repro (if code-related)

---

## License

MIT — see `LICENSE`. External datasets and models retain their original licenses.
