import streamlit as st
import PyPDF2
from docx import Document
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Skill list + normalization
skill_map = {
    "python": ["python"],
    "java": ["java"],
    "sql": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "javascript": ["javascript", "js"],
    "html": ["html"],
    "css": ["css"]
}

# -------------------------------
# 📄 Read PDF
def read_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text.lower()

# 📄 Read DOCX
def read_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text.lower()

# -------------------------------
# 🔍 Extract skills
def extract_skills(text):
    found = []
    for skill, variants in skill_map.items():
        for v in variants:
            if v in text:
                found.append(skill)
                break
    return list(set(found))

# -------------------------------
# 📊 TF-IDF Similarity
def compute_similarity(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return score

# -------------------------------
# 📈 Score Calculation
def calculate_score(jd_skills, resume_skills, similarity):

    # Skill score
    if len(jd_skills) == 0:
        skill_score = 0
    else:
        skill_score = len(set(jd_skills) & set(resume_skills)) / len(jd_skills)

    # Simplified scores
    experience_score = similarity
    education_score = 0.8
    keyword_score = similarity

    final_score = (
        skill_score * 0.5 +
        experience_score * 0.3 +
        education_score * 0.1 +
        keyword_score * 0.1
    ) * 100

    return round(final_score, 2)

# -------------------------------
# 💡 Recommendations
def give_recommendations(missing_skills):
    rec = []

    for skill in missing_skills:
        rec.append(f"Add experience or project related to {skill}")

    if not rec:
        rec.append("Good match! Try adding measurable achievements (numbers, results).")

    return rec

# -------------------------------
# 🌐 Streamlit UI
st.title("📄 AI Resume Screener (Beginner Version)")

jd = st.text_area("Enter Job Description")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX)")

if st.button("Analyze"):

    if jd and resume_file:

        start = time.time()

        # Read resume
        if resume_file.name.endswith(".pdf"):
            resume_text = read_pdf(resume_file)
        elif resume_file.name.endswith(".docx"):
            resume_text = read_docx(resume_file)
        else:
            st.error("Unsupported file format")
            st.stop()

        jd = jd.lower()

        # Extract skills
        jd_skills = extract_skills(jd)
        resume_skills = extract_skills(resume_text)

        # Similarity
        similarity = compute_similarity(jd, resume_text)

        # Score
        score = calculate_score(jd_skills, resume_skills, similarity)

        # Missing skills
        missing = list(set(jd_skills) - set(resume_skills))

        # Recommendations
        recs = give_recommendations(missing)

        end = time.time()

        # ---------------- OUTPUT ----------------
        st.subheader("📊 Results")

        st.write("Match Score:", score, "%")
        st.write("Similarity Score:", round(similarity, 2))

        st.write("✅ Matched Skills:", resume_skills)
        st.write("❌ Missing Skills:", missing)

        st.subheader("💡 Recommendations")
        for r in recs:
            st.write("-", r)

        st.write("⏱ Processing Time:", round(end - start, 2), "seconds")

    else:
        st.warning("Please enter Job Description and upload resume")