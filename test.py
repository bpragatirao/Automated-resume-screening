from pypdf import PdfReader
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import batch_to_device, cos_sim
import torch
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import requests

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    text = []

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text.append(t)
    return "\n".join(text)


# Load JD dataset
jd_df = pd.read_csv("C:/Users/engas/OneDrive/Desktop/ML MINI Project/Automated-resume-screening-main/data/raw/job_descriptions/job_dataset.csv").head(2)

resume_df = pd.read_csv("C:/Users/engas/OneDrive/Desktop/ML MINI Project/Automated-resume-screening-main/data/raw/resumes/Resume.csv").head(5)
# Convert JD to usable text (combine important fields)
jds = (
    jd_df["Title"].fillna('') + " " +
    jd_df["Skills"].fillna('') + " " +
    jd_df["Responsibilities"].fillna('')
).tolist()


# Load Resume dataset

resumes = resume_df["Resume_str"].fillna('').tolist()


def encode_batch(jobbert_model, texts):
    features = jobbert_model.tokenize(texts)
    features = batch_to_device(features, jobbert_model.device)
    features["text_keys"] = ["anchor"]
    with torch.no_grad():
        out_features = jobbert_model.forward(features)
    return out_features["sentence_embedding"].cpu().numpy()

def encode(jobbert_model, texts, batch_size: int = 8):
    # Sort texts by length and keep track of original indices
    sorted_indices = np.argsort([len(text) for text in texts])
    sorted_texts = [texts[i] for i in sorted_indices]
    
    embeddings = []
    
    # Encode in batches
    for i in tqdm(range(0, len(sorted_texts), batch_size)):
        batch = sorted_texts[i:i+batch_size]
        embeddings.append(encode_batch(jobbert_model, batch))
    
    # Concatenate embeddings and reorder to original indices
    sorted_embeddings = np.concatenate(embeddings)
    original_order = np.argsort(sorted_indices)
    return sorted_embeddings[original_order]

def compatibility_score(model: object, resume: list, jd: list)->list:
    job_embeddings = encode(model, jd)
    resume_embeddings = encode(model, resume)

    similarities = cos_sim(job_embeddings, resume_embeddings)
    adj_matrix = ((similarities + 1) / 2)*100

    return adj_matrix.tolist()

# def fetch_feedback(resume: object, jd: object, score: int):
#     prompt = f"list of resume content with: {resume} and job description: {jd} has similarity score of {score}% . can you provide reasoning behind this evaluation so that applicant understands the shorcomings in a concise manner? Point should clearly explain what's lacking from Job requirement to applicants resume, no assumptions should be made. Use higher weightage to technical skill requirement from job description while formulating your response, only 3 bullet points for each resume and jd"
#     # print(prompt)
#     payload = {
#             "model": "gemma:2b",
#             "prompt": prompt,
#             "stream": False
#             }

#     url = "http://localhost:11434/api/generate"
#     response = requests.post(url, json=payload)
#     print(response.json()["response"])
#     return response.json()["response"]

def fetch_feedback(resume: str, jd: str, score: float):
    decision = "SELECTED" if score >= 55 else "REJECTED"

    prompt = f"""
    Score: {score:.2f}%.

    Give ONLY ONE LINE:
    - Start with {decision}:
    - If selected → why good match
    - If rejected → what key technical skill is missing

    Rules:
    - Max 15 words
    - No assumptions
    - Focus on skills only
    """

    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False
    }

    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=payload)

    return response.json()["response"]
if __name__ == '__main__':
    
    model = SentenceTransformer("TechWolf/JobBERT-v2")

    # job_desc = pdf_to_text("data/job_descriptions/jd1.txt")
    # #
    # resume_desc_1 = pdf_to_text("data/resumes/full_stack_1.pdf")
    # resume_desc_2 = pdf_to_text("data/resumes/sde_1.pdf")
    # resume_desc_3 = pdf_to_text("data/resumes/jakes-resume.pdf")
    
    jd1 = None
    jd2 = None
    res1 = None
    res2 = None
    res3 = None
    with open ('data/job_descriptions/jd1.txt', 'r') as f:
        jd1 = f.read()
    with open ('data/job_descriptions/jd2.txt', 'r') as f:
        jd2 = f.read()
    with open ('data/resumes/resume1.txt', 'r') as f:
        res1 = f.read()
    with open ('data/resumes/resume2.txt', 'r') as f:
        res2 = f.read()
    with open ('data/resumes/resume3.txt', 'r') as f:
        res3 = f.read()

    
    # Note: score is in the form of correlation matrix
    scores = compatibility_score(model, resumes, jds)
    print(scores)

    # Note(anb): only fetched for resume with lower score
    # fetch_feedback(res1, jd1, scores)
    scores = compatibility_score(model, resumes, jds)

# scores shape: [len(jds)][len(resumes)]

    results = []

    for i, jd in enumerate(jds):
        print(f"\n========== JD {i+1} ==========")

        for j, resume in enumerate(resumes):
            score = scores[i][j]

            feedback = fetch_feedback(resume, jd, score)

            print(f"Resume {j+1} | Score: {score:.2f}%")
            print("Feedback:", feedback)

            # store results (important for UI / DB)
            results.append({
                "jd_id": i,
                "resume_id": j,
                "score": score,
                "feedback": feedback
            })
