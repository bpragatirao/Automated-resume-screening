import re
from pypdf import PdfReader
import easyocr
import os

_easy_ocr_reader = easyocr.Reader(['en'])

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

def ocr_to_text(path):
    text = _easy_ocr_reader.readtext(path, detail=0)
    return "".join(text)

def extract_text(path)->str:
    ext = os.path.splitext(path)[1].lower()
    result = "" 
    if ext == ".pdf":
        result = pdf_to_text(path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        result = ocr_to_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return clean_text(result)



