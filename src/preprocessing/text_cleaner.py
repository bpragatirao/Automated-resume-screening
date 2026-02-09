import re

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
