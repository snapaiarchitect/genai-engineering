import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re
from collections import Counter

st.set_page_config(page_title="Congressional Document Analysis", layout="wide")

st.title("🏛️ Congressional Document NLP Dashboard")
st.markdown("496 House Concurrent Resolutions from [Congress.gov](https://congress.gov)")

@st.cache_data
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "congress_bills.csv")
    df['latest_action_date'] = pd.to_datetime(df['latest_action_date'], errors='coerce')
    df['title_length'] = df['title'].str.len()
    return df

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bills", len(df))
col2.metric("House-Originated", f"{(df['origin_chamber'] == 'House').sum()}")
col3.metric("Median Title Length", f"{df['title_length'].median():.0f}")
col4.metric("Referred to Committee", f"{(df['latest_action_text'].str.contains('Referred', na=False)).sum()}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Bill Types")
    type_counts = df['type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(type_counts.index, type_counts.values, color=sns.color_palette('viridis', len(type_counts)))
    ax.set_xlabel('Count')
    st.pyplot(fig)

with col_right:
    st.subheader("Title Length Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['title_length'], bins=40, color='coral', edgecolor='white')
    ax.axvline(df['title_length'].median(), color='red', linestyle='--', label=f'Median: {df["title_length"].median():.0f}')
    ax.legend()
    ax.set_xlabel('Characters')
    st.pyplot(fig)

st.subheader("Top Keywords in Bill Titles")
all_words = []
for title in df['title'].dropna():
    words = re.findall(r'\b[A-Za-z]{4,}\b', title.lower())
    all_words.extend(words)
stop_words = {'bill', 'act', 'resolution', 'concurrent', 'joint', 'sense', 'congress', 'united', 'states', 'federal', 'national', 'shall', 'section'}
filtered = [w for w in all_words if w not in stop_words]
word_counts = Counter(filtered).most_common(15)
if word_counts:
    words, counts = zip(*word_counts)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(words)), counts, color=sns.color_palette('Spectral', len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    st.pyplot(fig)

st.subheader("Legislative Activity Timeline")
monthly = df[df['latest_action_date'].notna()].set_index('latest_action_date').resample('ME').size()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly.index, monthly.values, marker='o', color='steelblue')
ax.fill_between(monthly.index, monthly.values, alpha=0.3, color='steelblue')
ax.set_ylabel('Actions per Month')
st.pyplot(fig)

st.markdown("---")
st.caption("Data: Congress.gov API v3 | Fetched: 2026-05-21 | DEMO_KEY (development access)")
