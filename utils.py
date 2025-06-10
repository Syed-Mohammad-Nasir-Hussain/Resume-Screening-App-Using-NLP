import spacy
from sentence_transformers import SentenceTransformer, util

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def match_resume_to_job(resume_text, job_description):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_job = model.encode(job_description, convert_to_tensor=True)
    similarity = util.cos_sim(emb_resume, emb_job)
    return float(similarity[0][0])
