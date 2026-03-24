# import numpy as np
# import pandas as pd
# from tqdm.auto import tqdm
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import batch_to_device, cos_sim
# import torch
# import json
# import os
# import requests

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# # ================= LOAD DATA =================

# jd_df = pd.read_csv(
#     "C:/Users/engas/OneDrive/Desktop/ML MINI Project/Automated-resume-screening-main/data/raw/job_descriptions/job_dataset.csv"
# ).sample(5)

# resume_df = pd.read_csv(
#     "C:/Users/engas/OneDrive/Desktop/ML MINI Project/Automated-resume-screening-main/data/raw/resumes/Resume.csv"
# ).sample(100)


# # Prepare JD text for embedding
# jds = (
#     jd_df["Title"].fillna('') + " " +
#     jd_df["Skills"].fillna('') + " " +
#     jd_df["Responsibilities"].fillna('')
# ).tolist()

# # Prepare Resume text
# resumes = resume_df["Resume_str"].fillna('').tolist()


# # ================= ENCODING =================

# def encode_batch(model, texts):
#     features = model.tokenize(texts)
#     features = batch_to_device(features, model.device)
#     features["text_keys"] = ["anchor"]

#     with torch.no_grad():
#         out = model.forward(features)

#     return out["sentence_embedding"].cpu().numpy()


# def encode(model, texts, batch_size=16):
#     sorted_idx = np.argsort([len(t) for t in texts])
#     sorted_texts = [texts[i] for i in sorted_idx]

#     embeddings = []

#     for i in tqdm(range(0, len(sorted_texts), batch_size)):
#         batch = sorted_texts[i:i+batch_size]
#         embeddings.append(encode_batch(model, batch))

#     embeddings = np.concatenate(embeddings)
#     original_order = np.argsort(sorted_idx)

#     return embeddings[original_order]


# # ================= SIMILARITY =================

# def compatibility_score(model, resumes, jds):
#     job_emb = encode(model, jds)
#     res_emb = encode(model, resumes)

#     sim = cos_sim(job_emb, res_emb)
#     scores = ((sim + 1) / 2) * 100

#     return scores.tolist()


# # ================= FEEDBACK =================

# def fetch_feedback(resume, jd_title, jd_exp, jd_skills, jd_resp, resume_category, score):
    
#     prompt = f"""
# Job Description:
# Title: {jd_title}
# Experience Level: {jd_exp}
# Required Skills: {jd_skills}
# Key Responsibilities: {jd_resp}

# Candidate Resume:
# Category: {resume_category}
# Resume Content: {resume[:700]}

# Similarity Score: {score:.2f}%

# Task:
# Provide ONE concise feedback line (max 30 words).

# Instructions:
# - Mention 2–3 matching technical skills
# - Mention 1–2 missing skills
# - No generic phrases
# - Focus only on technical/domain skills

# Example:
# "C#, SQL, ASP.NET matched; missing Azure and REST API experience."
# """

#     payload = {
#         "model": "gemma:2b",
#         "prompt": prompt,
#         "stream": False
#     }

#     response = requests.post("http://localhost:11434/api/generate", json=payload)
#     return response.json()["response"]


# # ================= MAIN =================

# if __name__ == "__main__":

#     model = SentenceTransformer("TechWolf/JobBERT-v2")

#     print("🔄 Calculating similarity...")
#     scores = compatibility_score(model, resumes, jds)
#     print(scores)
#     TOP_K = 10

#     results = {}

#     for i in range(len(jds)):
#         print(f"\nProcessing JD {i+1}")

#         jd_scores = scores[i]

#         # Sort & pick top K
#         top_indices = np.argsort(jd_scores)[-TOP_K:][::-1]

#         candidates = []

#         for rank, idx in enumerate(top_indices, start=1):

#             score = jd_scores[idx]

#             feedback = fetch_feedback(
#                 resume=resumes[idx],
#                 jd_title=jd_df.iloc[i]["Title"],
#                 jd_exp=jd_df.iloc[i]["ExperienceLevel"],
#                 jd_skills=jd_df.iloc[i]["Skills"],
#                 jd_resp=jd_df.iloc[i]["Responsibilities"],
#                 resume_category=resume_df.iloc[idx]["Category"],
#                 score=score
#             )

#             candidates.append({
#                 "resume_id": int(idx),
#                 "category": resume_df.iloc[idx]["Category"],
#                 "score": float(score),
#                 "rank": rank,
#                 "feedback": feedback,
#                 "resume_preview": resumes[idx][:200]
#             })

#         results[f"jd_{i}"] = {
#             "jd_info": {
#                 "job_id": jd_df.iloc[i]["JobID"],
#                 "title": jd_df.iloc[i]["Title"],
#                 "experience": jd_df.iloc[i]["ExperienceLevel"]
#             },
#             "candidates": candidates
#         }


#     # ================= SAVE JSON =================

#     with open("results.json", "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)

#     print("✅ results.json created successfully!")

import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device, cos_sim
import torch
import requests
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, render_template, request

import sys
import os


# ================= ENCODING =================

def encode_batch(model, texts):
    features = model.tokenize(texts)
    features = batch_to_device(features, model.device)
    features["text_keys"] = ["anchor"]

    with torch.no_grad():
        out = model.forward(features)

    return out["sentence_embedding"].cpu().numpy()


def encode(model, texts, batch_size=16):
    sorted_idx = np.argsort([len(t) for t in texts])
    sorted_texts = [texts[i] for i in sorted_idx]

    embeddings = []

    for i in tqdm(range(0, len(sorted_texts), batch_size)):
        batch = sorted_texts[i:i+batch_size]
        embeddings.append(encode_batch(model, batch))

    embeddings = np.concatenate(embeddings)
    original_order = np.argsort(sorted_idx)

    return embeddings[original_order]


# ================= SIMILARITY =================

def compatibility_score(model, resumes, jds):
    job_emb = encode(model, jds)
    res_emb = encode(model, resumes)

    sim = cos_sim(job_emb, res_emb)
    scores = ((sim + 1) / 2) * 100

    return scores.tolist()


# ================= FEEDBACK =================

def fetch_feedback(resume_text, jd_text, score):
    prompt = f"""
Job Description:
{jd_text[:500]}

Candidate Resume:
{resume_text[:700]}

Similarity Score: {score:.2f}%

Task:
Give ONE short feedback (max 30 words)

Instructions:
- Mention 2–3 matching technical skills
- Mention 1–2 missing skills
- No generic wording
- Be specific

Example:
"Python, SQL, ML matched; missing Docker and Kubernetes."
"""

    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        return response.json().get("response", "").strip()
    except:
        return "Feedback generation failed"


# ================= MAIN PIPELINE =================

def process_data(model, resumes, jds, top_k=10):
    """
    resumes: list[str]
    jds: list[str]
    returns structured result dict
    """

    print("🔄 Calculating similarity...")

    scores = compatibility_score(model, resumes, jds)

    results = {}

    for i in range(len(jds)):
        print(f"Processing JD {i+1}")

        jd_scores = scores[i]

        # Top-K sorting
        top_indices = sorted(
            range(len(jd_scores)),
            key=lambda x: jd_scores[x],
            reverse=True
        )[:top_k]

        candidates = []

        for rank, idx in enumerate(top_indices, start=1):
            score = jd_scores[idx]

            feedback = fetch_feedback(
                resumes[idx],
                jds[i],
                score
            )

            candidates.append({
                "resume_id": idx,
                "rank": rank,
                "score": float(score),
                "feedback": feedback,
                "resume_preview": resumes[idx][:200]
            })

        results[f"jd_{i}"] = {
            "jd_text": jds[i][:300],
            "candidates": candidates
        }

    return results