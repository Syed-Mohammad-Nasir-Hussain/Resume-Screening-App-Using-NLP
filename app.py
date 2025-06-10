import streamlit as st
from PyPDF2 import PdfReader
from utils import extract_entities, match_resume_to_job
import pandas as pd

st.set_page_config(page_title="Resume Screening App", layout="wide")
st.title("📄 Resume Screening App using NLP")

mode = st.radio("🛠️ Choose Mode", ["Single Resume Match", "Multi Resume Comparison"])
job_description = st.text_area("📌 Paste the Job Description:")

def display_skill_tags(entities):
    skills = [ent for ent, label in entities if label in ['SKILL', 'ORG', 'PERSON']]
    top_skills = skills[:10]
    if top_skills:
        st.markdown("#### 🔖 Extracted Tags:")
        tag_html = " ".join([
            f"<span style='background-color:#e0f7fa; color:#006064; padding:4px 8px; margin:3px; border-radius:10px; display:inline-block;'>{skill}</span>"
            for skill in top_skills
        ])
        st.markdown(tag_html, unsafe_allow_html=True)

if mode == "Single Resume Match":
    uploaded_file = st.file_uploader("📤 Upload a Resume (PDF)", type=["pdf"])

    if uploaded_file and job_description:
        # Extract resume text safely
        resume_text = ""
        for page in PdfReader(uploaded_file).pages:
            page_text = page.extract_text()
            if page_text:
                resume_text += page_text

        st.write("🧾 Extracted resume text preview (first 500 chars):")
        st.write(resume_text[:500])

        if not resume_text.strip():
            st.error("❗ Could not extract any text from the resume. Please try another file.")
        else:
            score = match_resume_to_job(resume_text, job_description)
            entities = extract_entities(resume_text)

            st.subheader("✅ Matching Score")
            st.success(f"Score: {score:.2f} (Closer to 1.0 means better match)")

            st.subheader("📋 Extracted Entities")
            for ent, label in entities:
                st.markdown(f"- **{label}**: {ent}")

            display_skill_tags(entities)

elif mode == "Multi Resume Comparison":
    uploaded_files = st.file_uploader("📤 Upload Multiple Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        results = []
        for uploaded_file in uploaded_files:
            resume_text = ""
            for page in PdfReader(uploaded_file).pages:
                page_text = page.extract_text()
                if page_text:
                    resume_text += page_text

            if not resume_text.strip():
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}. Skipping.")
                continue

            score = match_resume_to_job(resume_text, job_description)
            entities = extract_entities(resume_text)
            top_entities = ', '.join([ent for ent, label in entities if label in ['SKILL', 'ORG', 'PERSON']][:5])
            results.append({
                "Resume File": uploaded_file.name,
                "Similarity Score": round(score, 2),
                "Top Entities": top_entities
            })

        if results:
            df = pd.DataFrame(results).sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)
            st.subheader("🏆 Ranked Resume Matches")
            st.dataframe(df.style.highlight_max(axis=0, subset=["Similarity Score"], color="lightgreen"))
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv, file_name='resume_match_results.csv', mime='text/csv')
        else:
            st.error("❗ No valid resumes processed. Please upload valid PDF resumes.")

else:
    st.info("Please upload resume(s) and enter the job description to get started.")
