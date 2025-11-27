import streamlit as st
from transformers import pipeline
import pandas as pd

# Load the fill-mask model only once
@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="bert-base-uncased")

fill_mask = load_model()

# App configuration
st.set_page_config(page_title="FillAI – Smart Text Completion", layout="centered")

st.title("FillAI – Smart Text Completion")
st.write("Enter a sentence containing **[MASK]** and the model will predict the missing word.")

# User input
text = st.text_input("Enter your sentence:", "The capital of India is [MASK].")
top_k = st.slider("How many predictions do you want?", 1, 10, 5)

# Predict button
if st.button("Predict"):
    if "[MASK]" not in text:
        st.error("Please include a [MASK] token in your sentence.")
    else:
        st.info("Predicting... please wait.")
        results = fill_mask(text, top_k=top_k)

        # Prepare table
        rows = []
        for item in results:
            rows.append({
                "Predicted Word": item["token_str"].strip(),
                "Probability": round(float(item["score"]), 4)
            })

        df = pd.DataFrame(rows)

        st.subheader("Top Predictions")
        st.table(df)
