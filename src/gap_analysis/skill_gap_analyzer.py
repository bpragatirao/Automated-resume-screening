def skill_gap(jd_skills, resume_skills):
    missing = set(jd_skills) - set(resume_skills)
    return list(missing)

