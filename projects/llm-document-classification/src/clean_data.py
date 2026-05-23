"""
Text Cleaning Pipeline
======================
Standard NLP preprocessing for document classification.
Source citations: Documents originate from ArXiv, PubMed, and Wikipedia live APIs.
"""

import re
import string
from pathlib import Path
from typing import List

import pandas as pd


RAW_PATH = Path("data/raw/all_documents.jsonl")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Apply standard NLP cleaning steps."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove LaTeX commands (common in ArXiv)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)  # inline math
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: List[str], stopwords: set = None) -> List[str]:
    """Remove English stopwords."""
    if stopwords is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stopwords = ENGLISH_STOP_WORDS
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.split()


def clean_documents(input_path: Path = RAW_PATH, output_path: Path = None) -> pd.DataFrame:
    """Load raw documents, clean text, and save processed DataFrame."""
    df = pd.read_json(input_path, lines=True)
    print(f"[clean_data] Loaded {len(df)} raw documents")

    df["clean_text"] = df["text"].apply(clean_text)
    # Remove empty documents
    df = df[df["clean_text"].str.len() > 20].reset_index(drop=True)
    print(f"[clean_data] {len(df)} documents after cleaning")

    if output_path is None:
        output_path = PROCESSED_DIR / "cleaned_documents.parquet"
    df.to_parquet(output_path, index=False)
    print(f"[clean_data] Saved to {output_path}")
    return df


if __name__ == "__main__":
    clean_documents()
