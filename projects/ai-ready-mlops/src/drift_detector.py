#!/usr/bin/env python3
"""
drift_detector.py — Statistical drift detection on real feature distributions.

Uses Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI)
to compare current input distributions against training baseline.

Alert thresholds:
- PSI < 0.1     : stable (green)
- PSI 0.1–0.25  : warning (yellow)
- PSI > 0.25    : critical (red)

Usage:
    python src/drift_detector.py
    # Or import: from src.drift_detector import DriftDetector
"""

import json
import sqlite3
import pickle
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.model_registry import ModelRegistry


BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"
DRIFT_DIR = DATA_DIR / "drift"
DRIFT_DIR.mkdir(exist_ok=True)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index between two distributions."""
    # Create bins from expected distribution
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Add small epsilon to avoid division by zero
    expected_pct = (expected_counts + 1e-6) / (expected_counts.sum() + 1e-6 * bins)
    actual_pct = (actual_counts + 1e-6) / (actual_counts.sum() + 1e-6 * bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test. Returns (statistic, p_value)."""
    stat, p_value = ks_2samp(expected, actual)
    return float(stat), float(p_value)


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    elif psi <= 0.25:
        return "warning"
    else:
        return "critical"


def classify_ks(p_value: float) -> str:
    return "drift" if p_value < 0.05 else "stable"


class DriftDetector:
    """Drift detection against production model baselines."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def get_production_baseline(self, experiment_name: str) -> Dict[str, Any]:
        """Load baseline distribution from production model."""
        run = self.registry.get_production_run(experiment_name)
        if not run:
            raise ValueError(f"No production model for {experiment_name}")
        baseline = self.registry.load_json(run["id"], "baseline_distribution.json")
        metadata = self.registry.load_json(run["id"], "metadata.json")
        return {
            "run_id": run["id"],
            "version": run["version"],
            "baseline": baseline,
            "feature_names": metadata.get("feature_names", []),
        }

    def detect_drift_for_experiment(
        self,
        experiment_name: str,
        current_data: pd.DataFrame,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Run KS + PSI for each feature against production baseline."""
        baseline_info = self.get_production_baseline(experiment_name)
        baseline = baseline_info["baseline"]

        results = {
            "experiment": experiment_name,
            "baseline_version": baseline_info["version"],
            "baseline_run_id": baseline_info["run_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "features": {},
            "overall_alert": "stable",
        }

        for feature in feature_names:
            if feature not in baseline or feature not in current_data.columns:
                continue

            # Generate expected distribution from baseline stats (mean, std)
            # In production this would be the actual training distribution saved as array
            mean = baseline[feature]["mean"]
            std = baseline[feature]["std"]
            expected = np.random.normal(mean, std, 1000)
            actual = current_data[feature].dropna().values

            if len(actual) == 0:
                continue

            psi = calculate_psi(expected, actual)
            ks_stat, ks_p = ks_test(expected, actual)

            feature_result = {
                "psi": round(psi, 4),
                "psi_status": classify_psi(psi),
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(ks_p, 4),
                "ks_status": classify_ks(ks_p),
                "expected_mean": round(float(mean), 4),
                "expected_std": round(float(std), 4),
                "actual_mean": round(float(np.mean(actual)), 4),
                "actual_std": round(float(np.std(actual)), 4),
            }
            results["features"][feature] = feature_result

            # Elevate overall alert if any feature is critical
            if feature_result["psi_status"] == "critical" or feature_result["ks_status"] == "drift":
                results["overall_alert"] = "critical"
            elif feature_result["psi_status"] == "warning" and results["overall_alert"] != "critical":
                results["overall_alert"] = "warning"

        # Save drift report
        report_path = DRIFT_DIR / f"{experiment_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Drift] Report saved: {report_path}")

        return results


def fetch_current_census_data() -> pd.DataFrame:
    """Fetch current Census ACS data for drift detection."""
    import json
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
    url = f"https://api.census.gov/data/2022/acs/acs5?get={var_list}&for=state:*"
    resp = urllib.request.urlopen(url, timeout=60)
    data = json.loads(resp.read())
    df = pd.DataFrame(data[1:], columns=data[0])
    numeric_cols = ["population", "median_income", "bachelors_degree", "total_pop_25_plus",
                    "median_age", "population_poverty", "population_poverty_total"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["education_rate"] = df["bachelors_degree"] / df["total_pop_25_plus"]
    df["poverty_rate"] = df["population_poverty"] / df["population_poverty_total"]
    return df.dropna(subset=["median_income"])


def fetch_current_bls_data() -> pd.DataFrame:
    """Fetch current BLS data for drift detection."""
    import json
    series = ["LNS14000000", "LNS11300000"]
    payload = json.dumps({
        "seriesid": series,
        "startyear": "2023",
        "endyear": "2024",
    }).encode()
    headers = {"Content-Type": "application/json"}
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
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
    df = df.pivot_table(index=["year", "period"], columns="series_id", values="value").reset_index()
    df.columns.name = None
    df = df.rename(columns={"LNS14000000": "unemployment_rate", "LNS11300000": "labor_force_participation"})
    df = df.sort_values(["year", "period"]).reset_index(drop=True)
    df["ur_lag1"] = df["unemployment_rate"].shift(1)
    df["ur_lag2"] = df["unemployment_rate"].shift(2)
    df["lfp_lag1"] = df["labor_force_participation"].shift(1)
    return df.dropna()


def fetch_current_arxiv_data() -> pd.DataFrame:
    """Fetch current arXiv abstracts for drift detection."""
    categories = ["cs.LG", "cs.AI", "cs.CL"]
    all_papers = []
    for cat in categories:
        query = urllib.parse.quote(f"cat:{cat}")
        url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            summary = entry.find("atom:summary", ns)
            all_papers.append({
                "abstract": (summary.text or "").strip() if summary is not None else "",
            })
    return pd.DataFrame(all_papers)


def main():
    print("🔍 Drift Detection — Running on real data")
    reg = ModelRegistry(str(DATA_DIR / "registry.db"))
    detector = DriftDetector(reg)

    # ── Census Drift Detection ──
    try:
        print("\n[Census] Detecting drift...")
        df = fetch_current_census_data()
        result = detector.detect_drift_for_experiment(
            "census-demographics", df,
            ["population", "median_age", "education_rate", "poverty_rate"]
        )
        print(f"[Census] Overall alert: {result['overall_alert']}")
        for feat, res in result["features"].items():
            print(f"  {feat}: PSI={res['psi']} ({res['psi_status']}), KS p={res['ks_p_value']} ({res['ks_status']})")
    except Exception as e:
        print(f"[Census] Drift detection failed: {e}")

    # ── BLS Drift Detection ──
    try:
        print("\n[BLS] Detecting drift...")
        df = fetch_current_bls_data()
        result = detector.detect_drift_for_experiment(
            "bls-employment", df,
            ["ur_lag1", "ur_lag2", "lfp_lag1"]
        )
        print(f"[BLS] Overall alert: {result['overall_alert']}")
        for feat, res in result["features"].items():
            print(f"  {feat}: PSI={res['psi']} ({res['psi_status']}), KS p={res['ks_p_value']} ({res['ks_status']})")
    except Exception as e:
        print(f"[BLS] Drift detection failed: {e}")

    print("\nDrift detection complete.")


if __name__ == "__main__":
    main()
