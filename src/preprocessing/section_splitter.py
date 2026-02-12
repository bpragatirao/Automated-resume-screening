def split_sections(text: str) -> dict:
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }

    for section in sections:
        if section in text:
            sections[section] = text.split(section, 1)[1][:600]

    return sections
