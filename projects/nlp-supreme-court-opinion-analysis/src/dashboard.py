import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from pathlib import Path
import re

st.set_page_config(page_title="SCOTUS Case Analysis", layout="wide")

st.title("⚖️ SCOTUS Case Outcome & Justice Voting Analysis")
st.markdown("59 decided cases from the 2022 Term | Oyez API")

@st.cache_data
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "scotus_cases.csv")
    df['date_argued'] = pd.to_datetime(df['date_argued'], errors='coerce')
    df['date_decided'] = pd.to_datetime(df['date_decided'], errors='coerce')
    df['decision_days'] = (df['date_decided'] - df['date_argued']).dt.days
    return df

df = load_data()

vote_cols = [c for c in df.columns if c.startswith('vote_')]
justices = [c.replace('vote_', '').replace('_', ' ') for c in vote_cols]

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", len(df))
col2.metric("Unanimous (9-0)", f"{(df['majority_votes'] == 9).sum()}")
col3.metric("Median Vote Margin", f"{(df['majority_votes'] - df['minority_votes']).median():.0f}")
col4.metric("Avg Decision Time", f"{df['decision_days'].median():.0f} days")

st.divider()

# Justice alignment matrix
st.subheader("Justice Voting Alignment Matrix")
alignment = pd.DataFrame(0, index=justices, columns=justices, dtype=float)
for _, case in df.iterrows():
    for j1 in justices:
        for j2 in justices:
            if j1 == j2:
                continue
            v1 = case.get(f'vote_{j1.replace(" ", "_")}', '')
            v2 = case.get(f'vote_{j2.replace(" ", "_")}', '')
            if v1 and v2 and v1 == v2:
                alignment.loc[j1, j2] += 1

for j1 in justices:
    for j2 in justices:
        if j1 != j2:
            both_present = df[[f'vote_{j1.replace(" ", "_")}', f'vote_{j2.replace(" ", "_")}']].notna().all(axis=1).sum()
            if both_present > 0:
                alignment.loc[j1, j2] = alignment.loc[j1, j2] / both_present

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(alignment, dtype=bool))
sns.heatmap(alignment, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0, vmax=1, square=True, ax=ax)
st.pyplot(fig)

# Vote margins and timeline
st.subheader("Case Characteristics")
col_left, col_right = st.columns(2)

with col_left:
    margins = df['majority_votes'] - df['minority_votes']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(margins, bins=range(0, 10), color='steelblue', edgecolor='white')
    ax.set_xlabel('Vote Margin (Majority - Minority)')
    ax.set_ylabel('Number of Cases')
    st.pyplot(fig)

with col_right:
    timeline = df[df['decision_days'].notna() & (df['decision_days'] > 0)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(timeline['decision_days'], bins=25, color='coral', edgecolor='white')
    ax.set_xlabel('Days from Argument to Decision')
    ax.set_ylabel('Number of Cases')
    st.pyplot(fig)

# Majority vote distribution
st.subheader("Majority Vote Distribution")
vote_counts = df['majority_votes'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(vote_counts.index, vote_counts.values, color=sns.color_palette('viridis', len(vote_counts)))
for x, y in zip(vote_counts.index, vote_counts.values):
    ax.text(x, y + 0.3, str(y), ha='center')
ax.set_xlabel('Majority Votes')
ax.set_ylabel('Number of Cases')
st.pyplot(fig)

st.markdown("---")
st.caption("Data: Oyez API | Term: 2022 | Fetched: 2026-05-21")
