from pypdf import PdfReader
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import batch_to_device, cos_sim
import torch

import requests

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    text = []

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text.append(t)

    return "\n".join(text)


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

    # sim = util.cos_sim(emb_resume, emb_jd).item()
    # score = (sim + 1) / 2
    similarities = cos_sim(job_embeddings, resume_embeddings)
    adj_matrix = ((similarities + 1) / 2)*100

    return adj_matrix.tolist()

def fetch_feedback(resume: object, jd: object, score: int)->None:
    prompt = f"resume content with: {resume} and job description: {jd} has similarity score of {score}% . provide reasoning behind this evaluation in a single liner. sentence should be formal which is directly displayed to user. response should not sound like a response to a request but simply a normal one liner (don't use sure, here's response or anything similar)"

    payload = {
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
            }

    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=payload)

    return response.json()["response"]


# if __name__ == '__main__':
#     model = SentenceTransformer("TechWolf/JobBERT-v3", local_files_only=True)
#
#     job_desc = pdf_to_text("job_requirement/COO Internship JD - MTech Intern.pdf")
#
#     resume_desc_1 = pdf_to_text("resume/full_stack_1.pdf")
#     resume_desc_2 = pdf_to_text("resume/sde_1.pdf")
#     resume_desc_3 = pdf_to_text("resume/jakes-resume.pdf")
#
#     scores = compatibility_score(model, [resume_desc_1, resume_desc_2, resume_desc_3], [job_desc])
#     print(scores)
#
#     # Note(anb): only fetched for resume with lower score
#     fetch_feedback(resume_desc_2, job_desc, scores[0][1])
