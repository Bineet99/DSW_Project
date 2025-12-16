import streamlit as st
import pandas as pd

# Try importing transformers + torch safely
try:
    from transformers import pipeline
    import torch  # checks if PyTorch exists
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    ERROR_MESSAGE = str(e)

st.set_page_config(page_title="FillAI – Smart Text Completion", layout="centered")

st.title("FillAI – Smart Text Completion")

# If PyTorch not installed → show friendly message
if not TORCH_AVAILABLE:
    st.error("Model backend not found. Please install PyTorch.\n\nError: " + ERROR_MESSAGE)
    st.stop()

# Load model once
@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="bert-base-uncased")

fill_mask = load_model()

st.write("Enter a sentence containing **[MASK]** and the model will predict the missing word.")

# User input
text = st.text_input("Enter your sentence:",
                     "He went to the [MASK] to buy groceries.”)
top_k = st.slider("How many predictions do you want?", 1, 10, 5)

if st.button("Predict"):
    if "[MASK]" not in text:
        st.error("Please include a [MASK] token in your sentence.")
    else:
        st.info("Predicting... please wait.")
        results = fill_mask(text, top_k=top_k)

        rows = []
        for item in results:
            rows.append({
                "Predicted Word": item["token_str"].strip(),
                "Probability": round(float(item["score"]), 4)
            })

        df = pd.DataFrame(rows)

        st.subheader("Top Predictions")
        st.table(df)

