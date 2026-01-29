from src.preprocessing.resume_parser import extract_text_from_pdf

resume_path = "data/resumes/resume1.pdf"

text = extract_text_from_pdf(resume_path)

print("----- EXTRACTED TEXT START -----")
print(text[:2000])   # print only first 2000 chars
print("----- EXTRACTED TEXT END -----")

#Done 