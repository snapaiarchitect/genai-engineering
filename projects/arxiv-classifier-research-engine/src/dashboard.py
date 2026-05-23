import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import json

st.set_page_config(page_title="arXiv Classifier Dashboard", layout="wide")

st.title("📚 arXiv Classifier & Research Engine Dashboard")
st.markdown("Real-time arXiv API data | 500 papers | NLP classification pipeline")

@st.cache_data
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "arxiv_papers.csv")
    df['published'] = pd.to_datetime(df['published'], errors='coerce')
    return df

df = load_data()

# Load metadata if available
try:
    with open(Path(__file__).parent.parent / "data" / "arxiv_metadata.json") as f:
        meta = json.load(f)
    st.sidebar.metric("Total Abstracts", meta.get("total_records", len(df)))
    st.sidebar.metric("Categories", len(meta.get("categories", [])))
except:
    pass

col1, col2, col3 = st.columns(3)
col1.metric("Total Papers", len(df))
col2.metric("Unique Categories", df['primary_category'].nunique())
col3.metric("Date Range", f"{df['published'].min().strftime('%Y-%m-%d') if df['published'].notna().any() else 'N/A'} → {df['published'].max().strftime('%Y-%m-%d') if df['published'].notna().any() else 'N/A'}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Primary Category Distribution")
    cat_counts = df['primary_category'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette('viridis', len(cat_counts))
    ax.barh(cat_counts.index, cat_counts.values, color=colors)
    ax.set_xlabel('Number of Papers')
    st.pyplot(fig)

with col_right:
    st.subheader("Publication Timeline")
    monthly = df[df['published'].notna()].set_index('published').resample('ME').size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(monthly.index, monthly.values, marker='o', color='steelblue')
    ax.fill_between(monthly.index, monthly.values, alpha=0.3, color='steelblue')
    ax.set_ylabel('Papers per Month')
    st.pyplot(fig)

st.subheader("Abstract Length Distribution")
df['abstract_length'] = df['abstract'].str.len()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['abstract_length'], bins=40, color='coral', edgecolor='white')
ax.axvline(df['abstract_length'].median(), color='red', linestyle='--', label=f'Median: {df["abstract_length"].median():.0f}')
ax.set_xlabel('Abstract Length (characters)')
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.caption("Data: arXiv API | Fetched: Live | Categories: cs.LG, cs.CL, cs.AI, cs.CV, stat.ML")
