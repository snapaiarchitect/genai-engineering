"""
Feature Engineering
===================
Build TF-IDF vectors and optional sentence embeddings for document classification.
Source data: ArXiv, PubMed, Wikipedia live API corpus.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_tfidf_features(df: pd.DataFrame, max_features: int = 5000, ngram_range=(1, 2)):
    """Create TF-IDF feature matrix from cleaned text."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    return X, vectorizer


def encode_labels(df: pd.DataFrame, label_col: str = "category"):
    """Encode string labels to integers."""
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    return y, le


def build_features(input_path: Path = PROCESSED_DIR / "cleaned_documents.parquet",
                   output_dir: Path = PROCESSED_DIR):
    """End-to-end feature engineering pipeline."""
    df = pd.read_parquet(input_path)
    print(f"[feature_engineering] Loaded {len(df)} documents")

    X, vectorizer = build_tfidf_features(df)
    y, le = encode_labels(df)

    # Save artifacts
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.pkl")
    joblib.dump(le, output_dir / "label_encoder.pkl")
    np.save(output_dir / "X_tfidf.npy", X.toarray())
    np.save(output_dir / "y.npy", y)

    # Save mapping for reference
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    pd.DataFrame(list(mapping.items()), columns=["category", "label"]).to_csv(
        output_dir / "label_mapping.csv", index=False
    )

    print(f"[feature_engineering] TF-IDF shape: {X.shape}")
    print(f"[feature_engineering] Classes: {le.classes_}")
    print(f"[feature_engineering] Saved artifacts to {output_dir}")
    return X, y, vectorizer, le


if __name__ == "__main__":
    build_features()
