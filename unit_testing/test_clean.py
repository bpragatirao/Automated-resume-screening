from src.preprocessing.resume_parser import extract_text_from_pdf
from src.preprocessing.text_cleaner import clean_text

resume_path = "data/resumes/resume_1.pdf"

raw_text = extract_text_from_pdf(resume_path)
cleaned_text = clean_text(raw_text)

print("----- CLEANED TEXT SAMPLE -----")
print(cleaned_text[:1500])
print("----- END -----")

#Done