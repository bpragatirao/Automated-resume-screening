def split_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }

    for key in sections:
        if key in text:
            sections[key] = text.split(key)[1][:500]

    return sections
