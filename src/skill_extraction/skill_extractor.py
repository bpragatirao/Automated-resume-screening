import json
import requests

def extract_skills(text, skills_file):
    with open(skills_file) as f:
        skills = json.load(f)

    found = [skill for skill in skills if skill.lower() in text]
    return list(set(found))

def fetch_feedback(resume: object,jd: object, score: int):
    prompt = f"""
    Score: {score}%
    Task:
    Provide ONE concise feedback line (max 30 words).

    Instructions:
    - Mention 2–3 matching technical skills
    - Mention 1–2 missing skills
    - No generic phrases
    - Focus only on technical/domain skills

    Example:
    "C#, SQL, ASP.NET matched; missing Azure and REST API experience."
    """
    
    payload = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False
    }

    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=payload)
    return response.json()["response"]