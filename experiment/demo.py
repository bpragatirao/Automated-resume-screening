from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained model
# model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
model = SentenceTransformer("TechWolf/JobBERT-v2", device="mps")

def list_similarity(list1, list2)->float:
    # Encode strings
    emb1 = model.encode(list1, normalize_embeddings=True)
    emb2 = model.encode(list2, normalize_embeddings=True)

    # Aggregate (mean pooling)
    vec1 = np.mean(emb1, axis=0, keepdims=True)
    vec2 = np.mean(emb2, axis=0, keepdims=True)

    # Cosine similarity in [-1, 1]
    sim = cosine_similarity(vec1, vec2)[0][0]

    # Note(anb): (shift by one and then normalize)
    score = (sim + 1) / 2
    return float(score)

input_tokens = ["python", "machine learning", "data processing"]
requirements = ["ml", "python programming"]

score = list_similarity(input_tokens, requirements)
print(score)
