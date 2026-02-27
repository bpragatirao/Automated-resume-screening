from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("TechWolf/JobBERT-v2")

def get_embedding(text):
    return model.encode(text)
