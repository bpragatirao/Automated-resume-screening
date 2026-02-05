import os
import csv
from src.preprocessing.resume_parser import extract_text_from_pdf
from src.preprocessing.text_cleaner import clean_text
from src.skill_extraction.skill_extractor import extract_skills
from src.gap_analysis.skill_gap_analyzer import skill_gap

# 1️⃣ Load resume
pdf_path = "data/resumes/resume_1.pdf"
resume_text = clean_text(extract_text_from_pdf(pdf_path))

# 2️⃣ Load JD
jd_path = "data/job_descriptions/jd_1.txt"
jd_text = clean_text(open(jd_path, "r").read())

# 3️⃣ Skills file
skills_file = "data/skills/skills_list.json"

# 4️⃣ Extract skills
resume_skills = extract_skills(resume_text, skills_file)
jd_skills = extract_skills(jd_text, skills_file)

# 5️⃣ Compute missing skills
missing_skills = skill_gap(jd_skills, resume_skills)

# 6️⃣ Print report
print("===== Resume Skills =====")
print(resume_skills)
print("\n===== JD Skills =====")
print(jd_skills)
print("\n===== Missing Skills =====")
print(missing_skills)

# 7️⃣ Save to CSV
os.makedirs("results/skill_gap_reports", exist_ok=True)
csv_path = "results/skill_gap_reports/resume_1_gap.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Resume Skills", "JD Skills", "Missing Skills"])
    writer.writerow([", ".join(resume_skills),
                     ", ".join(jd_skills),
                     ", ".join(missing_skills)])

print(f"\nSkill gap report saved at: {csv_path}")
