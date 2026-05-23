"""
Source: arXiv API corpus + FAISS index
Streamlit dashboard: Search tab + Corpus Stats tab.
"""
import json
import os
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter

st.set_page_config(page_title="RAG Knowledge Base", layout="wide")

DATA_PATH = "data/raw/arxiv_corpus.json"


def load_corpus():
    if not os.path.exists(DATA_PATH):
        st.error("Corpus not found. Run: python src/download_corpus.py")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def corpus_stats(corpus):
    total = len(corpus)
    cats = Counter(c for r in corpus for c in r.get("categories", []))
    lengths = [len(r["summary"].split()) for r in corpus]
    df = pd.DataFrame({
        "Metric": ["Total Documents", "Unique Categories", "Avg Abstract Length (words)", "Max Abstract Length"],
        "Value": [total, len(cats), round(np.mean(lengths), 1), max(lengths)],
    })
    return df, cats


st.title("RAG Knowledge Base — Scientific Literature Search")

tab_search, tab_stats = st.tabs(["Search", "Corpus Stats"])

with tab_search:
    query = st.text_input("Ask a question:", "What is attention mechanism?")
    k = st.slider("Top-K", 1, 10, 5)
    if st.button("Run RAG"):
        with st.spinner("Running pipeline..."):
            import sys
            sys.path.insert(0, "src")
            from rag_pipeline import run
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    run(query, k=k)
                except Exception as e:
                    st.error(str(e))
            result = buf.getvalue()
        st.text_area("Answer", result, height=400)

with tab_stats:
    corpus = load_corpus()
    if corpus:
        df, cats = corpus_stats(corpus)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(pd.Series(dict(cats.most_common(15))))
