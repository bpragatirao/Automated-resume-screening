import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import batch_to_device, cos_sim
import numpy as np

model = SentenceTransformer("TechWolf/JobBERT-v2", local_files_only=True)

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

def compatibility_score(resume: list, jd: list)->list:
    job_embeddings = encode(model, jd)
    resume_embeddings = encode(model, resume)

    similarities = cos_sim(job_embeddings, resume_embeddings)
    # similarity value range between [-1, 1], clip it instead of scaling
    adj_matrix = np.maximum(0, similarities) * 100
    return adj_matrix.tolist()