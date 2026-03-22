# resume_text = clean_text(extract_text_from_pdf("data/resumes/resume_1.pdf"))
# jd_text = clean_text(open("data/job_descriptions/jd_1.txt").read())

import pandas as pd
from src.preprocessing.resume_parser import preprocess_resumes
from src.preprocessing.job_loader import preprocess_jobs
from src.preprocessing.text_cleaner import clean_text,pdf_to_text
from src.matching.similarity_calculator import compatibility_score
from pypdf import PdfReader
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import batch_to_device, cos_sim
import torch
import json
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import requests
model = SentenceTransformer("TechWolf/JobBERT-v2")

RESUME_INPUT = "data/raw/resumes/Resume.csv"
JOB_INPUT = "data/raw/job_descriptions/job_dataset.csv"

RESUME_OUTPUT = "data/processed/resumes_clean.csv"
JOB_OUTPUT = "data/processed/jobs_clean.csv"

# FAST DEV SETTINGS
RESUME_SAMPLE_SIZE = 1
JOB_SAMPLE_SIZE = 1

print(" Preprocessing resumes (limited)...")
resume_df = preprocess_resumes(
    RESUME_INPUT,
    RESUME_OUTPUT,
    sample_size=RESUME_SAMPLE_SIZE
)
print(f"Resumes used: {len(resume_df)}")

print("\n Preprocessing job descriptions (limited)...")
job_df = preprocess_jobs(
    JOB_INPUT,
    JOB_OUTPUT,
    sample_size=JOB_SAMPLE_SIZE
)
print(f"Jobs used: {len(job_df)}")

# resume_row = resume_df.iloc[0]
job_row = job_df.iloc[0]

# resume_text = resume_row["clean_text"]
jd_text = job_row["clean_description"]
results = []
TOP_K = 10

def fetch_feedback(resume: object,jd: object, score: int):
    prompt = f"""
    Score: {score}%
    Task:
    Provide ONE concise feedback line (max 30 words).

    Instructions:
    - Mention 2–3 matching technical skills
    - Mention 1–2 missing skills
    - No generic phrases
    - Focus only on technical/domain skills

    Example:
    "C#, SQL, ASP.NET matched; missing Azure and REST API experience."
    """
    
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False
    }

    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=payload)
    return response.json()["response"]

for idx, resume in resume_df.iterrows():
    resume_text = resume["clean_text"]
    score = compatibility_score(model,resume_text, jd_text)
    results.append({
        "resume_id": idx,
        "score": round(score[0][0], 4),
    })

scores_df = pd.DataFrame(results)
scores_df = scores_df.sort_values(by="score", ascending=False)

print("\n Resume:JD Match Score ")
print(scores_df.head(10).to_string(index=False))

# Note(anb): only fetched for resume with lower score
print(fetch_feedback(resume_text, jd_text, score[0][0]))