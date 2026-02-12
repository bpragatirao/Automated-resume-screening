import pandas as pd
from .text_cleaner import clean_text

def preprocess_jobs(
    input_path: str,
    output_path: str,
    sample_size: int = 50
):
    df = pd.read_csv(input_path)

    print("Job dataset columns:", df.columns.tolist())

    # limit dataset for fast testing
    df = df.head(sample_size)

    # ✅ Combine multiple fields to form JD text
    TEXT_FIELDS = [
        "Title",
        "Skills",
        "Responsibilities",
        "Keywords"
    ]

    # keep only fields that actually exist
    available_fields = [c for c in TEXT_FIELDS if c in df.columns]

    if not available_fields:
        raise ValueError("❌ No usable job text fields found!")

    # merge fields into a single string
    df["job_text"] = df[available_fields].fillna("").agg(" ".join, axis=1)

    # clean combined job text
    df["clean_description"] = df["job_text"].apply(clean_text)

    final_df = df[
        ["JobID", "Title", "ExperienceLevel", "YearsOfExperience", "clean_description"]
    ]

    final_df.to_csv(output_path, index=False)

    return final_df
