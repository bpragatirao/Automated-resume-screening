from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(res_emb, jd_emb):
    return cosine_similarity([res_emb], [jd_emb])[0][0]

df["similarity_score"] = df.apply(lambda row: get_similarity(row["resume_embedding"], row["jd_embedding"]), axis=1)
