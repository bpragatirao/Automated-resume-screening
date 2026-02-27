from src.preprocessing.resume_parser import extract_text_from_pdf
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.section_splitter import split_sections
from src.embeddings.sbert_embedder import get_embedding
from src.matching.similarity_calculator import calculate_similarity
from src.matching.section_weighting import weighted_score

# 1️⃣ Load resume and JD
print("Loading and cleaning resume...")
resume_text = clean_text(extract_text_from_pdf("data/resumes/resume_1.pdf"))

print("Loading and cleaning JD...")
jd_text = clean_text(open("data/job_descriptions/jd_1.txt").read())

# 2️⃣ Split into sections
resume_sections = split_sections(resume_text)
jd_sections = split_sections(jd_text)

# 3️⃣ Compute section-wise similarity
section_scores = {}

for section in resume_sections:
    print(f"Embedding section: {section} ...")
    if resume_sections[section] and jd_sections[section]:
        res_emb = get_embedding(resume_sections[section])
        jd_emb = get_embedding(jd_sections[section])
        section_scores[section] = calculate_similarity(res_emb, jd_emb)
    else:
        section_scores[section] = 0.0
    print(f"{section} done.")

# 4️⃣ Compute final weighted score
final_score = weighted_score(section_scores)

# 5️⃣ Print results
print("\n===== Section-wise Similarity Scores =====")
for section, score in section_scores.items():
    print(f"{section.capitalize()}: {score:.2f}")

print(f"\nFinal Resume–JD Weighted Match Score: {final_score:.2f}")
