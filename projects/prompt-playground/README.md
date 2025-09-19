# Prompt Playground

A lightweight, CPU-friendly Streamlit app for experimenting with prompt engineering using local FLAN-T5 models.

## Features

- **Local Model Inference**: Uses Google's FLAN-T5 models (small/base) for text generation
- **Interactive Interface**: Web-based UI with adjustable parameters
- **Prompt Engineering**: System prompts, few-shot examples, and user input
- **Parameter Control**: Temperature, max tokens, and seed settings
- **Experiment Logging**: Automatically logs all interactions to CSV

## Quick Start

### Prerequisites
```bash
pip install streamlit transformers torch pandas
```

### Run the App
```bash
cd projects/prompt-playground
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **System Prompt**: Set the instruction/context for the model
2. **Few-shot Examples**: Provide example Q&A pairs (optional)
3. **User Prompt**: Enter your question or request
4. **Parameters**:
   - **Temperature**: Controls randomness (0.0 = deterministic, 1.5 = very creative)
   - **Max New Tokens**: Maximum length of generated response
   - **Seed**: For reproducible results
5. **Generate**: Click to get the model's response

## Models Available

- **FLAN-T5-Small**: Faster, smaller model (~80MB)
- **FLAN-T5-Base**: Better quality, larger model (~250MB)

Both models run efficiently on CPU without requiring GPU acceleration.

## Experiment Logging

All interactions are automatically logged to `../../docs/evidence/prompt_log.csv` with:
- Timestamp
- Model used
- Parameters (temperature, max tokens)
- Full prompt (system + few-shot + user)
- Generated response

## Example Use Cases

### 1. Question Answering
```
System: You are a helpful AI assistant. Be concise.
User: What is the capital of France?
```

### 2. Few-shot Learning
```
System: Answer questions about programming.
Few-shot: Q: What is a variable?
A: A variable is a container that stores data values.

Q: What is a function?
A: A function is a reusable block of code that performs a specific task.
User: What is a loop?
```

### 3. Creative Writing
```
System: You are a creative writer. Write engaging short stories.
User: Write a story about a robot learning to paint.
```

## Technical Details

- **Framework**: Streamlit for the web interface
- **Models**: Hugging Face Transformers (FLAN-T5)
- **Caching**: Model loading is cached for performance
- **Logging**: Pandas DataFrame to CSV for experiment tracking

## Project Structure
```
prompt-playground/
├── app.py          # Main Streamlit application
├── README.md       # This file
└── requirements.txt # Dependencies (if needed)
```

## Part of AI Portfolio

This project demonstrates:
- Practical prompt engineering skills
- Local model deployment and inference
- Interactive AI application development
- Experiment tracking and documentation
- CPU-efficient model usage
