import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

st.set_page_config(page_title="Clinical Trial Landscape", layout="wide")

st.title("🏥 Clinical Trial Landscape Dashboard")
st.markdown("500 active cancer trials from [ClinicalTrials.gov](https://clinicaltrials.gov)")

@st.cache_data
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "clinical_trials.csv")
    df['enrollment_count'] = pd.to_numeric(df['enrollment_count'], errors='coerce')
    df['primary_condition'] = df['condition'].str.split(',').str[0].str.strip()
    return df

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trials", len(df))
col2.metric("Pharma Sponsored", f"{(df['sponsor_class'] == 'INDUSTRY').sum()}")
col3.metric("Multi-Site Trials", f"{(df['locations'] > 1).sum()}")
col4.metric("Median Enrollment", f"{df['enrollment_count'].median():.0f}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Trial Phase Distribution")
    phase_counts = df['phase'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette('viridis', len(phase_counts))
    ax.barh(phase_counts.index, phase_counts.values, color=colors)
    ax.set_xlabel('Number of Trials')
    st.pyplot(fig)

with col_right:
    st.subheader("Sponsor Type")
    sponsor_counts = df['sponsor_class'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = sns.color_palette('Set2', len(sponsor_counts))
    ax.pie(sponsor_counts.values, labels=sponsor_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    st.pyplot(fig)

st.subheader("Top Cancer Types by Trial Volume")
top_conditions = df['primary_condition'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('Spectral', len(top_conditions))
ax.barh(range(len(top_conditions)), top_conditions.values, color=colors)
ax.set_yticks(range(len(top_conditions)))
ax.set_yticklabels(top_conditions.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Trials')
st.pyplot(fig)

st.subheader("Enrollment vs Locations")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['locations'], df['enrollment_count'], 
                     alpha=0.5, c=pd.Categorical(df['sponsor_class']).codes, cmap='tab10')
ax.set_xlabel('Number of Locations')
ax.set_ylabel('Planned Enrollment')
ax.set_yscale('log')
st.pyplot(fig)

st.markdown("---")
st.caption("Data: ClinicalTrials.gov API v2 | Fetched: 2026-05-21 | 500 active cancer trials")
