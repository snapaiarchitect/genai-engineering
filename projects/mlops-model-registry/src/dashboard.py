import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
import os

st.set_page_config(page_title="MLOps Model Registry", layout="wide")

st.title("🔧 MLOps Model Registry Dashboard")
st.markdown("Census ACS 2022 — 3,222 U.S. Counties | Income Prediction Pipeline")

@st.cache_data
def load_data():
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "acs_county_data.csv")
    df = df[df['median_income'] > 0]
    return df

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties", len(df))
col2.metric("Median Income", f"${df['median_income'].median():,.0f}")
col3.metric("Max Income", f"${df['median_income'].max():,.0f}")
col4.metric("Min Income", f"${df['median_income'].min():,.0f}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Model Registry")
    registry_dir = Path(__file__).parent.parent / "models"
    models = ['random_forest_v1', 'ridge_v1']
    metrics = []
    for m in models:
        meta_path = registry_dir / f"{m}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            metrics.append({
                'model': m,
                'RMSE': f"${meta['metrics']['rmse']:,.0f}",
                'MAE': f"${meta['metrics']['mae']:,.0f}",
                'R²': f"{meta['metrics']['r2']:.3f}",
            })
    if metrics:
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)

with col_right:
    st.subheader("Income Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['median_income'], bins=50, color='steelblue', edgecolor='white')
    ax.axvline(df['median_income'].median(), color='red', linestyle='--', label=f'Median: ${df["median_income"].median():,.0f}')
    ax.set_xlabel('Median Household Income ($)')
    st.pyplot(fig)

st.subheader("Drift Detection (PSI)")
features = ['population', 'bachelors_degree', 'labor_force', 'median_rent', 'commute_time']
batch_a = df.sample(frac=0.5, random_state=42)
batch_b = df.drop(batch_a.index)

def calculate_psi(expected, actual, buckets=10):
    expected_percents = np.histogram(expected, bins=buckets, density=True)[0]
    actual_percents = np.histogram(actual, bins=buckets, density=True)[0]
    expected_percents = np.clip(expected_percents, 0.0001, 1)
    actual_percents = np.clip(actual_percents, 0.0001, 1)
    return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

psi_scores = {}
for feat in features:
    psi = calculate_psi(batch_a[feat].dropna(), batch_b[feat].dropna())
    psi_scores[feat] = psi

colors = ['seagreen' if v < 0.1 else 'gold' if v < 0.25 else 'crimson' for v in psi_scores.values()]
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(list(psi_scores.keys()), list(psi_scores.values()), color=colors)
ax.axvline(0.1, color='green', linestyle='--')
ax.axvline(0.25, color='red', linestyle='--')
ax.set_xlabel('PSI')
st.pyplot(fig)

st.markdown("---")
st.caption("Data: U.S. Census ACS 2022 | Fetched: 2026-05-21 | Models: Random Forest v1, Ridge v1")
