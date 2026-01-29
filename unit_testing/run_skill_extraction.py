from src.preprocessing.resume_parser import extract_text_from_pdf
from src.preprocessing.text_cleaner import clean_text
from src.skill_extraction.skill_extractor import extract_skills

# 1. Load resume PDF
pdf_path = "data/resumes/resume_1.pdf"
text = extract_text_from_pdf(pdf_path)

# 2. Clean text
cleaned_text = clean_text(text)

# 3. Extract skills
skills_file = "data/skills/skills_list.json"
found_skills = extract_skills(cleaned_text, skills_file)

# 4. Print skills
print("===== Skills Found in Resume =====")
for skill in found_skills:
    print("-", skill)
