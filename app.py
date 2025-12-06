import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import docx
from PyPDF2 import PdfReader

# -----------------------------------------------------------------------
# Download NLTK resources (safe for deployment)
# -----------------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# -----------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------
@st.cache_resource
def load_model():
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
        text += page.extract_text() or ""
    return text

# -----------------------------------------------------------------------
# Text Preprocessing
# -----------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------
st.title("📄 Resume Classification App")
st.write("Upload a resume (PDF or DOCX) to predict its category.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:

    # Extract text
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    st.subheader("Extracted Text")
    st.write(resume_text[:1000] + " ...")

    # Clean text
    cleaned = clean_text(resume_text)

    # Predict
    prediction = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Category: **{prediction}**")

    # Show probabilities
    st.subheader("Prediction Probabilities")
    df_proba = pd.DataFrame({
        "Category": model.classes_,
        "Probability": np.round(proba, 4)
    })

    st.dataframe(df_proba)
