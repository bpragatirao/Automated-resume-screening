import pandas as pd
from .text_cleaner import clean_text
from .section_splitter import split_sections

def preprocess_resumes(
    input_path: str,
    output_path: str,
    sample_size: int = 100
):
    df = pd.read_csv(input_path)

    # 🔍 Safety check (good practice)
    print("Resume columns:", df.columns.tolist())

    # limit dataset for fast testing
    df = df.head(sample_size)

    df["clean_text"] = df["Resume_str"].apply(clean_text)

    sections = df["clean_text"].apply(split_sections)
    sections_df = pd.json_normalize(sections)

    final_df = pd.concat(
        [df[["Category"]], sections_df, df["clean_text"]],
        axis=1
    )

    final_df.to_csv(output_path, index=False)

    TEXT_COLUMN_CANDIDATES = ["Resume", "Resume_str", "resume", "text"]

    resume_col = next(
        (c for c in TEXT_COLUMN_CANDIDATES if c in df.columns),
        None
    )

    if resume_col is None:
        raise ValueError("❌ No resume text column found!")

    df["clean_text"] = df[resume_col].apply(clean_text)

    return final_df
