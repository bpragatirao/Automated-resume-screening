from src.preprocessing.resume_parser import extract_text_from_pdf
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.section_splitter import split_sections

# 1. Load resume PDF
pdf_path = "data/resumes/resume_1.pdf"
text = extract_text_from_pdf(pdf_path)

# 2. Clean text
cleaned_text = clean_text(text)

# 3. Split into sections
sections = split_sections(cleaned_text)

# 4. Print sections
print("===== Resume Sections =====")
for section, content in sections.items():
    print(f"\n[{section.upper()}]:")
    print(content[:500])  # first 500 chars only

#done output->
# nguages cc javascript web html css reactjs bootstrap tailwind databases sql mysql developer tools vs code git github postman achievements qualified gate 2025 cse with all india rank 2678 among 175000 candidates leetcode coding streak of 200 days ranked in top 15 globally with contest rating 1675

# [EXPERIENCE]:
#  software engineering intern oct 2022  may 2023 consultadd services pvt ltd pune  implemented lazy loading in a reactjs ecommerce app cutting loading time to 450 ms  developed restful apis using spring boot gaining handson

# [EDUCATION]:
#  national institute of technology karnataka nitk 2025  present master of technology mtech in computer science gate rank 2678  175000 jawaharlal institute of technology khargone mp 2020  2024 bachelor of technology in information technology cgpa 8 experience software engineering intern oct 2022  may 2023 consultadd services pvt ltd pune  implemented lazy loading in a reactjs ecommerce app cutting loading time to 450 ms  developed restful apis using spring boot gaining handson experience with aws

# [PROJECTS]:
#   created and maintained an opensource web app hosting 50la