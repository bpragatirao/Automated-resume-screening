# resume_text = clean_text(extract_text_from_pdf("data/resumes/resume_1.pdf"))
# jd_text = clean_text(open("data/job_descriptions/jd_1.txt").read())

import pandas as pd

from src.preprocessing.resume_parser import preprocess_resumes
from src.preprocessing.job_loader import preprocess_jobs
from src.embeddings.sbert_embedder import get_embedding
from src.matching.similarity_calculator import calculate_similarity

RESUME_INPUT = "data/raw/resumes/Resume.csv"
JOB_INPUT = "data/raw/job_descriptions/job_dataset.csv"

RESUME_OUTPUT = "data/processed/resumes_clean.csv"
JOB_OUTPUT = "data/processed/jobs_clean.csv"

# 🔹 FAST DEV SETTINGS
RESUME_SAMPLE_SIZE = 100
JOB_SAMPLE_SIZE = 50

print("🔹 Preprocessing resumes (limited)...")
resume_df = preprocess_resumes(
    RESUME_INPUT,
    RESUME_OUTPUT,
    sample_size=RESUME_SAMPLE_SIZE
)
print(resume_df.head(2))
print(f"Resumes used: {len(resume_df)}")

print("\n🔹 Preprocessing job descriptions (limited)...")
job_df = preprocess_jobs(
    JOB_INPUT,
    JOB_OUTPUT,
    sample_size=JOB_SAMPLE_SIZE
)
print(job_df.head(2))
print(f"Jobs used: {len(job_df)}")

# resume_row = resumes_df.iloc[0]
job_row = job_df.iloc[0]

# resume_text = resume_row["clean_text"]
jd_text = job_row["clean_description"]

# resume_emb = get_embedding(resume_text)
jd_emb = get_embedding(jd_text)

results = []

for idx, resume in resume_df.iterrows():
    resume_text = resume["clean_text"]

    resume_emb = get_embedding(resume_text)
    score = calculate_similarity(resume_emb, jd_emb)

    results.append({
        "resume_id": idx,
        "category": resume["Category"],
        "job_title": job_row["Title"],
        "score": round(score, 4)
    })

scores_df = pd.DataFrame(results)
scores_df = scores_df.sort_values(by="score", ascending=False)

print("\n✅ Resume:JD Match Score ")
print(scores_df.head(10).to_string(index=False))
