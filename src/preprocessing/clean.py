import re

def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip().lower()

#df["clean_resume"] = df["resume_text"].apply(clean_text)
#df["clean_jd"] = df["job_description"].apply(clean_text)

