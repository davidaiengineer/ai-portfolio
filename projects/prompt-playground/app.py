import time
from pathlib import Path
import pandas as pd
import streamlit as st
import random

st.set_page_config(page_title="Prompt Playground", page_icon="✨", layout="centered")

# Check if transformers/PyTorch is available
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

def load_model_safe(model_name: str):
    """Safely load model only if transformers is available"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# UI Setup
st.title("✨ Prompt Playground")

if TRANSFORMERS_AVAILABLE:
    st.caption("🤖 AI-powered with FLAN-T5. Logs your prompt experiments to CSV.")
    st.success("✅ PyTorch detected - Real AI responses available!")
else:
    st.caption("🎭 Demo Mode - Interface fully functional, simulated AI responses.")
    st.warning("⚠️ PyTorch not detected - Running in demo mode. Install PyTorch for real AI responses.")

# Input Controls
colA, colB = st.columns([2,1])
with colA:
    system_prompt = st.text_area("System / Instruction", value="You are a helpful AI assistant. Be concise.")
with colB:
    if TRANSFORMERS_AVAILABLE:
        model_name = st.selectbox("Model", ["google/flan-t5-small","google/flan-t5-base"], index=1)
    else:
        model_name = st.selectbox("Model (Demo)", ["google/flan-t5-small (demo)","google/flan-t5-base (demo)"], index=1)

fewshot = st.text_area("Few-shot Examples (optional)", placeholder="Q: What is AI?\nA: AI is artificial intelligence.\n\nQ: What is ML?\nA: ML is machine learning.")
user_prompt = st.text_area("User Prompt", placeholder="Ask me anything...")

c1, c2, c3 = st.columns(3)
with c1:
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
with c2:
    max_new_tokens = st.slider("Max New Tokens", 16, 512, 192, 16)
with c3:
    seed = st.number_input("Seed", value=42, step=1)

run = st.button("🚀 Generate", type="primary")

# Generation Logic
if run and user_prompt.strip():
    with st.spinner("🤖 Generating response..."):
        # Build prompt
        prompt = ""
        if system_prompt.strip():
            prompt += f"Instruction:\n{system_prompt.strip()}\n\n"
        if fewshot.strip():
            prompt += f"Examples:\n{fewshot.strip()}\n\n"
        prompt += f"User:\n{user_prompt.strip()}\n\nAssistant:"

        # Generate response
        if TRANSFORMERS_AVAILABLE:
            # Real AI mode
            pipe = load_model_safe(model_name.replace(" (demo)", ""))
            if pipe:
                try:
                    result = pipe(
                        prompt,
                        do_sample=True,
                        temperature=float(temperature),
                        max_new_tokens=int(max_new_tokens),
                        eos_token_id=pipe.tokenizer.eos_token_id,
                        pad_token_id=pipe.tokenizer.eos_token_id
                    )
                    out = result[0]["generated_text"]
                except Exception as e:
                    out = f"Error generating response: {e}"
            else:
                out = "Failed to load model. Please check your setup."
        else:
            # Demo mode
            demo_responses = [
                f"Demo response: Based on your prompt '{user_prompt.strip()[:50]}...', I would provide a contextual answer using FLAN-T5.",
                f"🎭 This is a simulated response! Temperature: {temperature}, Max tokens: {max_new_tokens}. Install PyTorch for real AI.",
                f"Demo mode active. Your system prompt was: '{system_prompt[:30]}...'. Real mode would use this for context.",
                "This demonstrates the interface! The real model would analyze your few-shot examples and generate appropriate responses.",
                f"Simulated AI response with seed {seed}. The interface works perfectly - just need PyTorch for real generation!"
            ]
            out = random.choice(demo_responses)
            time.sleep(1)  # Simulate processing time

    # Display results
    st.subheader("📝 Response")
    if not TRANSFORMERS_AVAILABLE:
        st.info("🎯 This is a simulated response. Install PyTorch to enable real FLAN-T5 generation.")
    
    st.write(out)

    # Log to CSV
    try:
        log_dir = Path("../../docs/evidence")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "prompt_log.csv"
        
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "system_prompt": system_prompt,
            "fewshot_examples": fewshot,
            "user_prompt": user_prompt,
            "response": out,
            "mode": "real" if TRANSFORMERS_AVAILABLE else "demo"
        }
        
        df = pd.DataFrame([row])
        if log_path.exists():
            df.to_csv(log_path, mode="a", index=False, header=False)
        else:
            df.to_csv(log_path, index=False)
        
        st.success(f"✅ Logged to {log_path}")
    except Exception as e:
        st.error(f"Failed to log interaction: {e}")

elif run and not user_prompt.strip():
    st.warning("⚠️ Please enter a user prompt to generate a response.")

# Footer
st.markdown("---")
st.markdown("**💡 Tip:** Experiment with different temperatures (0.1 = focused, 1.0 = creative) and few-shot examples!")
if not TRANSFORMERS_AVAILABLE:
    st.markdown("**🔧 To enable real AI:** `pip install torch transformers` then restart the app.")