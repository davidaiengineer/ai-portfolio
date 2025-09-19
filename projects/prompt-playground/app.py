import time
from pathlib import Path
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="Prompt Playground", page_icon="✨", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

st.title("✨ Prompt Playground")
st.caption("Lightweight, local, CPU-friendly. Logs your prompt experiments to CSV.")

colA, colB = st.columns([2,1])
with colA:
    system_prompt = st.text_area("System / Instruction", value="You are a helpful AI assistant. Be concise.")
with colB:
    model_name = st.selectbox("Model", ["google/flan-t5-small","google/flan-t5-base"], index=1)

fewshot = st.text_area("Few-shot Examples (optional)", placeholder="Q: ...\nA: ...\n\nQ: ...\nA: ...")
user_prompt = st.text_area("User Prompt", placeholder="Ask me anything...")

c1, c2, c3 = st.columns(3)
with c1:
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
with c2:
    max_new_tokens = st.slider("Max New Tokens", 16, 512, 192, 16)
with c3:
    seed = st.number_input("Seed", value=42, step=1)

run = st.button("Generate")

if run:
    with st.spinner("Loading model and generating..."):
        pipe = load_model(model_name)
        prompt = ""
        if system_prompt.strip():
            prompt += f"Instruction:\n{system_prompt.strip()}\n\n"
        if fewshot.strip():
            prompt += f"Examples:\n{fewshot.strip()}\n\n"
        prompt += f"User:\n{user_prompt.strip()}\n\nAssistant:"

        out = pipe(
            prompt,
            do_sample=True,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.eos_token_id
        )[0]["generated_text"]

    st.subheader("Response")
    st.write(out)

    # Log to CSV
    log_dir = Path("../../docs/evidence")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "prompt_log.csv"
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "system": system_prompt,
        "fewshot": fewshot,
        "user": user_prompt,
        "response": out
    }
    df = pd.DataFrame([row])
    if log_path.exists():
        df.to_csv(log_path, mode="a", index=False, header=False)
    else:
        df.to_csv(log_path, index=False)
    st.success(f"Logged to {log_path}")
