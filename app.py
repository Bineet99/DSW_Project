from transformers import pipeline
import streamlit as st
import pandas as pd

# Load the fill-mask model only once
@st.cache_resource
def get_model():
    return pipeline("fill-mask", model="bert-base-uncased")

# Function to get predictions
def get_predictions(text, top_k):
    model = get_model()
    return model(text, top_k=top_k)

st.set_page_config(page_title="FillAI", layout="centered")

st.title("FillAI – Smart Text Completion")
st.write("Type a sentence that contains [MASK] and see the predicted words.")

# User input
text = st.text_input("Enter your sentence:", "The capital of India is [MASK].")
top_k = st.slider("How many suggestions do you want?", 1, 10, 5)

# Button to run prediction
if st.button("Predict"):
    if "[MASK]" not in text:
        st.error("Please add a [MASK] in your sentence.")
    else:
        st.info("Predicting, please wait…")
        results = get_predictions(text, top_k)

        # Prepare results table
        rows = []
        for item in results:
            rows.append({
                "Word": item["token_str"].strip(),
                "Probability": round(float(item["score"]), 4)
            })

        df = pd.DataFrame(rows)
        st.table(df)
