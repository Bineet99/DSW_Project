import streamlit as st
import pandas as pd
try:
    from transformers import pipeline
    import torch 
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    ERROR_MESSAGE = str(e)
st.set_page_config(page_title="FillAI – Smart Text Completion", layout="centered")
st.title("FillAI – Smart Text Completion")
if not TORCH_AVAILABLE:
    st.error("Model backend not found. Please install PyTorch.\n\nError: " + ERROR_MESSAGE)
    st.stop()
@st.cache_resource
def load_model():
    return pipeline("fill-mask", model="bert-base-uncased")
fill_mask = load_model()
st.write("Enter a sentence containing **[MASK]** and the model will predict the missing word.")
text = st.text_input(
    "Enter your sentence:",
    "He is reading a [MASK] in the library."
)
top_k = st.slider("How many predictions do you want?", 1, 10, 1)
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
    "Probability": f"{round(item['score'] * 100)}%"
})
        df = pd.DataFrame(rows)
        st.subheader("Top Predictions")
        st.table(df)
