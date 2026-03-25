import numpy as np


def skill_gap(jd_skills, resume_skills):
    missing = set(jd_skills) - set(resume_skills)
    return list(missing)


def top_n_resumes_per_jd(jds: list, resumes: list, score: list, limit: int):
    score = np.asarray(score)
    m, n = score.shape

    if m != len(jds):
        print("score and jd count mismatch")
        exit(1)

    if n != len(resumes):
        print("score and resume count mismatch")
        exit(1)

    limit = min(limit, n)
    results = []

    for i in range(m):
        row = score[i]
        sorted_idx = np.argsort(row)[::-1]

        top_set = set(sorted_idx[:limit])

        results.append({
            "jd": jds[i],
            "resumes": [
                {
                    "resume": resumes[j],
                    "score": row[j],
                    "index": j,
                    "label": "pass" if j in top_set else "fail"
                }
                for j in sorted_idx
            ]
        })

    return results