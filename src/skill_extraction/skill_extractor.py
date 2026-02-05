import json

def extract_skills(text, skills_file):
    with open(skills_file) as f:
        skills = json.load(f)

    found = [skill for skill in skills if skill.lower() in text]
    return list(set(found))
