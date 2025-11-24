import os
import streamlit as st
import pandas as pd
import requests

API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def call_hf_inference(text, top_k=5):
    payload = {"inputs": text, "parameters": {"top_k": top_k}}
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

st.set_page_config(page_title="FillAI", layout="centered")
st.title("FillAI – Smart Text Completion (remote inference)")
st.write("Type a sentence that contains [MASK] and see the predicted words.")

text = st.text_input("Enter your sentence:", "The capital of India is [MASK].")
top_k = st.slider("How many suggestions do you want?", 1, 10, 5)

if st.button("Predict"):
    if "[MASK]" not in text:
        st.error("Please add a [MASK] in your sentence.")
    else:
        try:
            with st.spinner("Querying model…"):
                results = call_hf_inference(text, top_k=top_k)
            # results is a list of dicts with token_str and score
            rows = []
            for item in results:
                rows.append({
                    "Word": item.get("token_str", "").strip(),
                    "Probability": round(float(item.get("score", 0)), 4)
                })
            df = pd.DataFrame(rows)
            st.table(df)
        except requests.exceptions.HTTPError as http_err:
            st.error(f"Inference API error: {http_err}")
            st.write(resp.text if 'resp' in locals() else "")
        except Exception as e:
            st.error("Something went wrong. Check logs.")
            st.write(str(e))
