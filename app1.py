import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import docx
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -----------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Make sure best_model.pkl is in the same folder as this app.py
    return joblib.load("best_model.pkl")

model = load_model()

# -----------------------------------------------------------------------
# Text Extraction Functions
# -----------------------------------------------------------------------
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -----------------------------------------------------------------------
# Text Preprocessing (NO NLTK)
# -----------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Simple tokenization
    tokens = text.split()

    # Remove stopwords and very short tokens
    tokens = [
        t for t in tokens
        if t not in ENGLISH_STOP_WORDS and len(t) > 2
    ]

    # Join back
    return " ".join(tokens)

# -----------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------
st.title("📄 Resume Classification App")
st.write("Upload a resume (PDF or DOCX) to predict its category.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    if not resume_text or resume_text.strip() == "":
        st.error("Could not extract any text from the file. Please upload a text-based PDF or DOCX.")
    else:
        st.subheader("Extracted Text (first 1000 characters)")
        st.write(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))

        # Clean text
        cleaned = clean_text(resume_text)

        if cleaned.strip() == "":
            st.error("After cleaning, no useful text remained for prediction.")
        else:
            # Predict
            try:
                prediction = model.predict([cleaned])[0]

                # If model supports predict_proba
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([cleaned])[0]

                st.subheader("Prediction Result")
                st.success(f"Predicted Category: **{prediction}**")

                if proba is not None:
                    st.subheader("Prediction Probabilities")
                    df_proba = pd.DataFrame({
                        "Category": model.classes_,
                        "Probability": np.round(proba, 4)
                    })
                    st.dataframe(df_proba)

            except Exception as e:
                st.error("Error while making prediction. Please check the model compatibility.")
                st.exception(e)
