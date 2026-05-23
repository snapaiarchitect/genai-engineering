import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="PubMed Research Engine", layout="wide")

st.title("🧬 PubMed Research Engine Dashboard")
st.markdown("NCBI E-utilities API | Medical/health abstracts | NLP pipeline")

@st.cache_data
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "pubmed_abstracts.csv")
    return df

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Total Abstracts", len(df))
col2.metric("Unique Journals", df['journal'].nunique() if 'journal' in df.columns else "N/A")
col3.metric("Avg Authors", f"{df['author_count'].mean():.1f}" if 'author_count' in df.columns else "N/A")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Publication Timeline")
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(year_counts.index.astype(str), year_counts.values, color='steelblue')
        ax.set_xlabel('Year')
        st.pyplot(fig)

with col_right:
    st.subheader("Top Journals")
    if 'journal' in df.columns:
        journal_counts = df['journal'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette('Set2', len(journal_counts))
        ax.barh(journal_counts.index, journal_counts.values, color=colors)
        ax.set_xlabel('Articles')
        st.pyplot(fig)

st.subheader("Abstract Length Distribution")
if 'abstract' in df.columns:
    df['abstract_length'] = df['abstract'].str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['abstract_length'], bins=40, color='coral', edgecolor='white')
    ax.axvline(df['abstract_length'].median(), color='red', linestyle='--', label=f'Median: {df["abstract_length"].median():.0f}')
    ax.set_xlabel('Characters')
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.caption("Data: PubMed E-utilities API | Fetched: 2026-05-21 | Query: machine learning")
