from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_emb, jd_emb):
    return cosine_similarity([resume_emb], [jd_emb])[0][0]

