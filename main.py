from src.preprocessing.resume_parser import extract_text_from_pdf
from src.preprocessing.text_cleaner import clean_text
from src.embeddings.sbert_embedder import get_embedding
from src.matching.similarity_calculator import calculate_similarity

resume_text = clean_text(extract_text_from_pdf("data/resumes/resume_1.pdf"))
jd_text = clean_text(open("data/job_descriptions/jd_1.txt").read())

resume_emb = get_embedding(resume_text)
jd_emb = get_embedding(jd_text)

score = calculate_similarity(resume_emb, jd_emb)
print("Resume–JD Match Score:", score)
