import re
from pypdf import PdfReader

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    text = []

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text.append(t)
    return "\n".join(text)
