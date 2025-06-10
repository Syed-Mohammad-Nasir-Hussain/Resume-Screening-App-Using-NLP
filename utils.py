import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Cache the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Cache the spaCy language model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Load models
model = load_model()
nlp = load_spacy_model()

# Extract named entities from text
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Compute similarity score between resume and job description
def match_resume_to_job(resume_text, job_description):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_job = model.encode(job_description, convert_to_tensor=True)
    similarity = util.cos_sim(emb_resume, emb_job)
    return float(similarity[0][0])

