import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import json

st.set_page_config(page_title="SCOTUS Opinion Mining", layout="wide")

st.title("⚖️ SCOTUS Opinion Mining Dashboard")
st.markdown("CourtListener API | Supreme Court opinions | NLP + topic modeling")

@st.cache_data
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "scotus_opinions.csv")
    df['date_filed'] = pd.to_datetime(df['date_filed'], errors='coerce')
    return df

df = load_data()

col1, col2, col3 = st.columns(3)
col1.metric("Total Opinions", len(df))
col2.metric("Unique Dockets", df['docket_number'].nunique() if 'docket_number' in df.columns else "N/A")
col3.metric("Date Range", f"{df['date_filed'].min().strftime('%Y-%m-%d') if 'date_filed' in df.columns and df['date_filed'].notna().any() else 'N/A'} → {df['date_filed'].max().strftime('%Y-%m-%d') if 'date_filed' in df.columns and df['date_filed'].notna().any() else 'N/A'}")

st.divider()

# Load topic data if available
try:
    with open(Path(__file__).parent.parent / "data" / "scotus_metadata.json") as f:
        meta = json.load(f)
    st.sidebar.metric("Topics Identified", meta.get("topic_count", "N/A"))
    st.sidebar.metric("Sentiment Range", f"{meta.get('sentiment_min', 'N/A')} to {meta.get('sentiment_max', 'N/A')}")
except:
    pass

st.subheader("Opinion Text Length Distribution")
if 'text' in df.columns:
    df['text_length'] = df['text'].str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['text_length'], bins=40, color='steelblue', edgecolor='white')
    ax.axvline(df['text_length'].median(), color='red', linestyle='--', label=f'Median: {df["text_length"].median():.0f}')
    ax.set_xlabel('Opinion Length (characters)')
    ax.legend()
    st.pyplot(fig)
else:
    st.info("No text data available")

st.markdown("---")
st.caption("Data: CourtListener API | Fetched: Live | Supreme Court opinions 2023–2026")
