"""
Streamlit Dashboard — Interactive Document Classification Demo
================================================================
Visualize corpus distribution, model predictions, and per-class metrics.

Run with: streamlit run dashboard.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Document Classification Dashboard", layout="wide")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

st.title("📄 Multi-Domain Document Classification")
st.markdown(
    "Live corpus from **ArXiv** (scientific/financial), **PubMed** (medical), "
    "and **Wikipedia** (legal/financial). TF-IDF + Random Forest baseline."
)

# Load data
try:
    df = pd.read_parquet(PROCESSED_DIR / "cleaned_documents.parquet")
    le = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    vectorizer = joblib.load(PROCESSED_DIR / "tfidf_vectorizer.pkl")
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    data_ready = True
except Exception as exc:
    st.error(f"Pipeline artifacts not found. Run `python src/run_pipeline.py` first.\n\n{exc}")
    data_ready = False

if data_ready:
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_source = st.sidebar.multiselect(
        "Source", options=df["source"].unique(), default=df["source"].unique()
    )
    selected_category = st.sidebar.multiselect(
        "Category", options=df["category"].unique(), default=df["category"].unique()
    )

    filtered = df[
        df["source"].isin(selected_source) & df["category"].isin(selected_category)
    ]

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", len(filtered))
    col2.metric("Categories", filtered["category"].nunique())
    col3.metric("Sources", filtered["source"].nunique())
    col4.metric("Avg Text Length", int(filtered["text"].str.len().mean()))

    # Category distribution
    st.subheader("Category Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=filtered, x="category", order=filtered["category"].value_counts().index, ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Source distribution
    st.subheader("Source Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    source_counts = filtered["source"].value_counts()
    ax2.pie(source_counts, labels=source_counts.index, autopct="%1.1f%%", startangle=140)
    ax2.set_title("Documents by Source")
    st.pyplot(fig2)

    # Model predictions on sample
    st.subheader("🔍 Live Prediction Demo")
    sample_idx = st.number_input("Document index", min_value=0, max_value=len(filtered) - 1, value=0)
    sample = filtered.iloc[sample_idx]
    st.write(f"**Title:** {sample['title']}")
    st.write(f"**True Category:** {sample['category']}  |  **Source:** {sample['source']}")
    with st.expander("Show full text"):
        st.write(sample["text"][:2000])

    vec = vectorizer.transform([sample["clean_text"]])
    pred_rf = le.inverse_transform([rf.predict(vec)[0]])[0]
    pred_proba_rf = rf.predict_proba(vec)[0]
    pred_lr = le.inverse_transform([lr.predict(vec)[0]])[0]
    pred_proba_lr = lr.predict_proba(vec)[0]

    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        st.metric("Random Forest Prediction", pred_rf)
        st.write({cls: f"{p:.2%}" for cls, p in zip(le.classes_, pred_proba_rf)})
    with col_pred2:
        st.metric("Logistic Regression Prediction", pred_lr)
        st.write({cls: f"{p:.2%}" for cls, p in zip(le.classes_, pred_proba_lr)})

    # Metrics
    st.subheader("📊 Model Metrics")
    for metrics_file in sorted(REPORTS_DIR.glob("*_metrics.json")):
        metrics = pd.read_json(metrics_file, typ="series")
        st.write(f"**{metrics['model']}** — Accuracy: `{metrics['accuracy']:.3f}`, F1-macro: `{metrics['f1_macro']:.3f}`")
        st.json(dict(metrics.get("per_class_f1", {})))

    # Confusion matrices
    st.subheader("Confusion Matrices")
    for cm_file in sorted(REPORTS_DIR.glob("*_confusion_matrix.png")):
        st.image(str(cm_file), caption=cm_file.stem.replace("_", " ").title())
