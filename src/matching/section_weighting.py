SECTION_WEIGHTS = {
    "skills": 0.4,
    "experience": 0.3,
    "projects": 0.2,
    "education": 0.1
}

def weighted_score(section_scores):
    return sum(section_scores[k] * SECTION_WEIGHTS[k] for k in section_scores)
