#!/usr/bin/env python3
"""
retrain_pipeline.py — Automated retraining with performance comparison.

Fetches fresh real data from Census, BLS, and arXiv APIs.
Retrains models, compares vs. previous production version,
auto-rejects if regression > 2%, registers new version if accepted.

Usage:
    python src/retrain_pipeline.py
    # Or import and call main()
"""

import sys
import json
import time
import pickle
import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from src.model_registry import ModelRegistry


BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. DATA FETCHERS — Real public data APIs
# ══════════════════════════════════════════════════════════════════════════

def fetch_census_acs() -> pd.DataFrame:
    """Fetch REAL ACS 5-Year state-level demographics from Census API.
    Source: https://api.census.gov/data/2022/acs/acs5
    Citation: U.S. Census Bureau (2024), American Community Survey 5-Year Estimates.
    """
    variables = {
        "NAME": "state_name",
        "B01003_001E": "population",
        "B19013_001E": "median_income",
        "B15003_022E": "bachelors_degree",
        "B15003_001E": "total_pop_25_plus",
        "B01002_001E": "median_age",
        "B17001_002E": "population_poverty",
        "B17001_001E": "population_poverty_total",
    }
    var_list = ",".join(variables.keys())
    key = os.environ.get("CENSUS_API_KEY", "")
    url = f"https://api.census.gov/data/2022/acs/acs5?get={var_list}&for=state:*&key={key}"
    print(f"[Census] Fetching ACS data... {url[:90]}")
    try:
        resp = urllib.request.urlopen(url, timeout=60)
        data = json.loads(resp.read())
        df = pd.DataFrame(data[1:], columns=data[0])
        # Rename columns from Census codes to readable names
        df = df.rename(columns=variables)
        # Convert numeric columns
        numeric_cols = ["population", "median_income", "bachelors_degree", "total_pop_25_plus",
                        "median_age", "population_poverty", "population_poverty_total"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["education_rate"] = df["bachelors_degree"] / df["total_pop_25_plus"]
        df["poverty_rate"] = df["population_poverty"] / df["population_poverty_total"]
        # Drop rows with missing income (common for PR/territories)
        df = df.dropna(subset=["median_income"])
        print(f"[Census] Fetched {len(df)} states/territories")
        return df
    except Exception as e:
        print(f"[Census] ERROR: {e}")
        raise


def fetch_bls_employment() -> pd.DataFrame:
    """Fetch REAL BLS employment time series via Public API.
    Source: https://api.bls.gov/publicAPI/v2/timeseries/data/
    Citation: U.S. Bureau of Labor Statistics (2024), Labor Force Statistics.
    """
    series = ["LNS14000000", "LNS11300000"]  # unemployment rate, labor force participation
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "seriesid": series,
        "startyear": "2019",
        "endyear": "2024",
    }).encode()
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    print(f"[BLS] Fetching employment series...")
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        if data.get("status") != "REQUEST_SUCCEEDED":
            print(f"[BLS] API status: {data.get('status')}")
            return pd.DataFrame()
        records = []
        for s in data.get("Results", {}).get("series", []):
            sid = s["seriesID"]
            for item in s.get("data", []):
                records.append({
                    "series_id": sid,
                    "year": int(item["year"]),
                    "period": item["period"],
                    "value": float(item["value"]),
                })
        df = pd.DataFrame(records)
        # Pivot to wide format
        df = df.pivot_table(index=["year", "period"], columns="series_id", values="value").reset_index()
        df.columns.name = None
        df = df.rename(columns={"LNS14000000": "unemployment_rate", "LNS11300000": "labor_force_participation"})
        df = df.sort_values(["year", "period"]).reset_index(drop=True)
        print(f"[BLS] Fetched {len(df)} monthly observations")
        return df
    except Exception as e:
        print(f"[BLS] ERROR: {e}")
        raise


def fetch_arxiv_abstracts(max_per_cat: int = 150) -> pd.DataFrame:
    """Fetch REAL arXiv abstracts via Atom API.
    Source: http://export.arxiv.org/api/query
    Citation: arXiv.org (2024), arXiv API.
    """
    categories = ["cs.LG", "cs.AI", "cs.CL"]
    all_papers = []
    print(f"[arXiv] Fetching abstracts...")
    for cat in categories:
        query = urllib.parse.quote(f"cat:{cat}")
        url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_per_cat}&sortBy=submittedDate&sortOrder=descending"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            root = ET.fromstring(data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                paper = {
                    "title": (title.text or "").strip() if title is not None else "",
                    "abstract": (summary.text or "").strip() if summary is not None else "",
                    "primary_category": cat,
                }
                all_papers.append(paper)
            time.sleep(0.5)  # be nice to arXiv
        except Exception as e:
            print(f"[arXiv] ERROR for {cat}: {e}")
    df = pd.DataFrame(all_papers)
    print(f"[arXiv] Fetched {len(df)} abstracts")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINERS
# ══════════════════════════════════════════════════════════════════════════

def train_census_classifier(df: pd.DataFrame) -> Dict[str, Any]:
    """Train Random Forest classifier on Census ACS data.
    Task: Predict high/low income bracket from demographics.
    """
    features = ["population", "median_age", "education_rate", "poverty_rate"]
    X = df[features].fillna(0)
    y = (df["median_income"] > df["median_income"].median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    # Cross-validation for robustness
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)))

    return {
        "model": model,
        "metrics": {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
        },
        "feature_names": features,
        "baseline_distribution": {f: {"mean": float(X[f].mean()), "std": float(X[f].std())} for f in features},
    }


def train_bls_forecaster(df: pd.DataFrame) -> Dict[str, Any]:
    """Train Random Forest regressor on BLS unemployment time series.
    Task: Predict next-month unemployment rate.
    """
    df = df.dropna(subset=["unemployment_rate", "labor_force_participation"])
    if len(df) < 6:
        raise ValueError("Not enough BLS data to train")
    # Lag features
    df = df.copy()
    df["ur_lag1"] = df["unemployment_rate"].shift(1)
    df["ur_lag2"] = df["unemployment_rate"].shift(2)
    df["lfp_lag1"] = df["labor_force_participation"].shift(1)
    df = df.dropna()

    features = ["ur_lag1", "ur_lag2", "lfp_lag1"]
    X = df[features]
    y = df["unemployment_rate"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.mean(np.abs(preds - y_test))

    return {
        "model": model,
        "metrics": {
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "test_mean_actual": round(float(y_test.mean()), 4),
        },
        "feature_names": features,
        "baseline_distribution": {f: {"mean": float(X[f].mean()), "std": float(X[f].std())} for f in features},
    }


def train_arxiv_classifier(df: pd.DataFrame) -> Dict[str, Any]:
    """Train TF-IDF + Logistic Regression on arXiv abstracts.
    Task: Multi-class classification of primary category.
    """
    df = df.dropna(subset=["abstract", "primary_category"])
    X_text = df["abstract"].astype(str)
    y = df["primary_category"]

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1, 4),
        },
        "feature_names": ["tfidf_abstract"],
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. RETRAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def retrain_and_register(
    reg: ModelRegistry,
    experiment_name: str,
    model_name: str,
    train_fn,
    fetch_fn,
    metric_for_comparison: str,
    regression_threshold: float = 0.02,
) -> Optional[int]:
    """Fetch fresh data, retrain, compare vs. production, register or reject."""
    print(f"\n{'='*60}")
    print(f"Retraining: {experiment_name}/{model_name}")
    print(f"{'='*60}")

    # Fetch fresh data
    df = fetch_fn()
    if len(df) == 0:
        print(f"[SKIP] No data fetched for {experiment_name}")
        return None

    # Train
    result = train_fn(df)
    new_metrics = result["metrics"]
    print(f"[NEW MODEL] Metrics: {json.dumps(new_metrics, indent=2)}")

    # Compare vs. production
    prod_run = reg.get_production_run(experiment_name)
    if prod_run:
        prod_metric = prod_run.get("metrics", {}).get(metric_for_comparison, 0)
        new_metric = new_metrics.get(metric_for_comparison, 0)
        print(f"[COMPARE] Production {metric_for_comparison}={prod_metric:.4f}, New={new_metric:.4f}")

        # Determine regression direction
        higher_is_better = metric_for_comparison in ("accuracy", "f1", "f1_weighted", "cv_mean")
        if higher_is_better:
            regression = prod_metric - new_metric
        else:
            regression = new_metric - prod_metric

        if regression > regression_threshold:
            print(f"[REJECTED] Regression={regression:.4f} > threshold={regression_threshold}")
            return None
        else:
            print(f"[ACCEPTED] Regression={regression:.4f} within threshold")
    else:
        print(f"[ACCEPTED] No production model found — first run")

    # Register new run
    params = {
        "n_estimators": 100,
        "random_state": 42,
        "train_samples": len(df),
    }
    run_id = reg.start_run(experiment_name, model_name, params)
    reg.log_metrics(run_id, new_metrics)

    # Save model artifact
    artifact_name = "model.pkl"
    if "vectorizer" in result:
        # For arXiv, save both model and vectorizer
        reg.save_model(run_id, {"model": result["model"], "vectorizer": result["vectorizer"]}, artifact_name)
    else:
        reg.save_model(run_id, result["model"], artifact_name)

    # Save baseline distribution for drift detection
    reg.save_json(run_id, result["baseline_distribution"], "baseline_distribution.json")
    reg.save_json(run_id, {"feature_names": result["feature_names"]}, "metadata.json")

    # Auto-promote if first run, else tag as staging
    if prod_run is None:
        reg.promote(experiment_name, run_id, promoted_by="retrain_pipeline", reason="First production model")
        print(f"[PROMOTED] run_id={run_id} to production")
    else:
        reg.set_tag(run_id, "stage", "staging")
        print(f"[STAGING] run_id={run_id} tagged for A/B test")

    return run_id


def main():
    print("🔄 AI-Ready MLOps — Retrain Pipeline")
    print(f"Started at: {datetime.utcnow().isoformat()}")

    reg = ModelRegistry(str(DATA_DIR / "registry.db"))

    # ── Model A: Census Demographics Classifier ──
    try:
        retrain_and_register(
            reg, "census-demographics", "census_classifier",
            train_census_classifier, fetch_census_acs,
            metric_for_comparison="accuracy", regression_threshold=0.02,
        )
    except Exception as e:
        print(f"[ERROR] Census retrain failed: {e}")

    # ── Model B: BLS Employment Forecaster ──
    try:
        retrain_and_register(
            reg, "bls-employment", "bls_forecaster",
            train_bls_forecaster, fetch_bls_employment,
            metric_for_comparison="rmse", regression_threshold=0.02,
        )
    except Exception as e:
        print(f"[ERROR] BLS retrain failed: {e}")

    # ── Model C: arXiv Abstract Classifier ──
    try:
        retrain_and_register(
            reg, "arxiv-abstracts", "arxiv_classifier",
            train_arxiv_classifier, fetch_arxiv_abstracts,
            metric_for_comparison="accuracy", regression_threshold=0.02,
        )
    except Exception as e:
        print(f"[ERROR] arXiv retrain failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    for exp in reg.list_experiments():
        prod = reg.get_production_run(exp["name"])
        print(f"  {exp['name']}: production={prod['version'] if prod else 'None'}")
        runs = reg.list_runs(exp["name"])
        print(f"    Total runs: {len(runs)}")

    print(f"\nFinished at: {datetime.utcnow().isoformat()}")
    return reg


if __name__ == "__main__":
    main()
