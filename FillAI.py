import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

try:
    from transformers import pipeline
    import torch
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    ERROR_MESSAGE = str(e)

st.set_page_config(page_title="FillAI – Smart Text Completion", layout="centered")

st.title("FillAI – Smart Text Completion")
st.write("Predict missing words using a BERT language model.")

if not TORCH_AVAILABLE:
    st.error("PyTorch not installed.\n\nError: " + ERROR_MESSAGE)
    st.stop()

@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="bert-base-uncased")

fill_mask = load_model()

text = st.text_input(
    "Enter sentence with [MASK]",
    "He is reading a [MASK] in the library."
)

top_k = st.slider("Number of predictions", 1, 10, 5)

if st.button("Predict"):

    if "[MASK]" not in text:
        st.error("Please include [MASK] token.")
    else:
        results = fill_mask(text, top_k=top_k)

        words = []
        scores = []

        for item in results:
            word = item["token_str"].strip()
            score = round(item["score"] * 100, 2)

            words.append(word)
            scores.append(score)

            st.write(f"{word} ({score}%)")

        df = pd.DataFrame({
            "Word": words,
            "Probability": scores
        })

        st.subheader("Prediction Table")
        st.table(df)

        st.subheader("Prediction Chart")

        fig, ax = plt.subplots()
        ax.barh(words, scores)
        ax.set_xlabel("Probability (%)")
        ax.set_ylabel("Predicted Word")

        st.pyplot(fig)
