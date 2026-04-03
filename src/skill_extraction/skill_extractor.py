import json
import requests

def extract_skills(text, skills_file):
    with open(skills_file) as f:
        skills = json.load(f)

    found = [skill for skill in skills if skill.lower() in text]
    return list(set(found))

def fetch_feedback(resume: object,jd: object, score: int):
    prompt = f"""
    Task:
    Provide ONE concise feedback line (max 30 words).

    Instructions:
    - Mention 2–3 matching technical skills
    - Mention 1–2 missing skills
    - No generic phrases
    - Focus only on technical/domain skills
    - response should be in professional tone.

    Here's the Info:
    Score: {score}
    Job Description: {jd}
    Resume: {resume}
    """
    
    payload = {
        "model": "gemma4:e4b",
        "prompt": prompt,
        "stream": False
    }

    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=payload)
    return response.json()["response"]
