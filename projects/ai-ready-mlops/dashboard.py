#!/usr/bin/env python3
"""
dashboard.py — Streamlit monitoring dashboard.

Run: streamlit run dashboard.py

Shows:
- Model registry timeline
- Drift detection alerts
- A/B test results
- Inference monitoring (latency, cost, accuracy)
"""

import json
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

from src.model_registry import ModelRegistry
from src.monitor import InferenceMonitor
from src.drift_detector import DriftDetector, classify_psi


BASE = Path(__file__).parent
DATA_DIR = BASE / "data"

st.set_page_config(page_title="AI-Ready MLOps Dashboard", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("🧠 AI-Ready MLOps")
st.sidebar.markdown("Production ML Model Lifecycle Management")

reg = ModelRegistry(str(DATA_DIR / "registry.db"))
experiments = [e["name"] for e in reg.list_experiments()]
selected_exp = st.sidebar.selectbox("Select Experiment", experiments + ["All"], index=len(experiments))

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🔧 MLOps Monitoring Dashboard")
st.markdown(f"Last updated: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC`")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📦 Registry", "🔍 Drift", "🧪 A/B Tests", "📊 Monitoring"])

# ── Tab 1: Model Registry ──────────────────────────────────────────────────
with tab1:
    st.header("Model Registry Timeline")
    if not experiments:
        st.info("No experiments found. Run the retrain pipeline first.")
    else:
        for exp_name in ([selected_exp] if selected_exp != "All" else experiments):
            with st.expander(f"📁 {exp_name}", expanded=True):
                prod = reg.get_production_run(exp_name)
                if prod:
                    st.success(f"🏷️ Production: v{prod['version']} (run {prod['id']})")
                    cols = st.columns(3)
                    for i, (k, v) in enumerate(prod.get("metrics", {}).items()):
                        cols[i % 3].metric(k, round(v, 4) if isinstance(v, float) else v)

                runs = reg.list_runs(exp_name)
                if runs:
                    df = pd.DataFrame([
                        {
                            "Run ID": r["id"],
                            "Version": r["version"],
                            "Created": r["created_at"][:19] if r["created_at"] else "",
                            "Stage": r.get("tags", {}).get("stage", "unknown"),
                        }
                        for r in runs
                    ])
                    st.dataframe(df, use_container_width=True)

                # Promotion history
                history = reg.get_promotion_history(exp_name)
                if history:
                    st.markdown("**Promotion History**")
                    hist_df = pd.DataFrame([
                        {
                            "Time": h["promoted_at"][:19],
                            "From Run": h["from_run_id"],
                            "To Run": h["to_run_id"],
                            "By": h["promoted_by"],
                            "Reason": h["reason"],
                        }
                        for h in history[:10]
                    ])
                    st.dataframe(hist_df, use_container_width=True)

# ── Tab 2: Drift Detection ─────────────────────────────────────────────────
with tab2:
    st.header("Drift Detection Alerts")
    DRIFT_DIR = DATA_DIR / "drift"
    drift_files = sorted(DRIFT_DIR.glob("*.json"), reverse=True) if DRIFT_DIR.exists() else []

    if not drift_files:
        st.info("No drift reports found. Run `python src/drift_detector.py` first.")
    else:
        # Show latest report per experiment
        shown_exps = set()
        for f in drift_files:
            with open(f) as fp:
                report = json.load(fp)
            exp = report.get("experiment")
            if selected_exp != "All" and exp != selected_exp:
                continue
            if exp in shown_exps:
                continue
            shown_exps.add(exp)

            alert_color = {"stable": "🟢", "warning": "🟡", "critical": "🔴"}.get(
                report.get("overall_alert", "stable"), "⚪"
            )
            with st.expander(f"{alert_color} {exp} — Baseline: v{report.get('baseline_version')}", expanded=True):
                st.text(f"Report time: {report.get('timestamp', 'unknown')}")
                feat_data = []
                for feat, res in report.get("features", {}).items():
                    feat_data.append({
                        "Feature": feat,
                        "PSI": res["psi"],
                        "PSI Status": res["psi_status"],
                        "KS p-value": res["ks_p_value"],
                        "KS Status": res["ks_status"],
                        "Expected μ": res["expected_mean"],
                        "Actual μ": res["actual_mean"],
                    })
                if feat_data:
                    st.dataframe(pd.DataFrame(feat_data), use_container_width=True)

# ── Tab 3: A/B Test Results ────────────────────────────────────────────────
with tab3:
    st.header("A/B Test Results")
    st.info("Run `python src/ab_test.py` to generate A/B test results.")
    # Show simulated summary if we have experiments
    if experiments:
        for exp_name in ([selected_exp] if selected_exp != "All" else experiments):
            try:
                prod = reg.get_production_run(exp_name)
                if not prod:
                    continue
                runs = reg.list_runs(exp_name)
                candidates = [r for r in runs if r.get("id") != prod.get("id")]
                if not candidates:
                    continue
                cand = candidates[0]
                with st.expander(f"🧪 {exp_name}", expanded=True):
                    c1, c2 = st.columns(2)
                    c1.metric("Control (Production)", f"v{prod['version']}")
                    c2.metric("Candidate (Staging)", f"v{cand['version']}")
                    st.markdown("*Simulated performance data — run `python src/ab_test.py` for live results*")
            except Exception:
                pass

# ── Tab 4: Monitoring ──────────────────────────────────────────────────────
with tab4:
    st.header("Inference Monitoring")
    monitor = InferenceMonitor()

    if not experiments:
        st.info("No experiments found.")
    else:
        for exp_name in ([selected_exp] if selected_exp != "All" else experiments):
            latency = monitor.get_latency_percentiles(exp_name, window_hours=24)
            cost = monitor.estimate_cost(exp_name)

            with st.expander(f"📊 {exp_name}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("p50 Latency", f"{latency['p50']} ms")
                c2.metric("p95 Latency", f"{latency['p95']} ms")
                c3.metric("p99 Latency", f"{latency['p99']} ms")
                c4.metric("24h Inferences", latency["count"])

                c5, c6 = st.columns(2)
                c5.metric("Mean Latency", f"{latency['mean']} ms")
                c6.metric("Est. 24h Cost", f"${cost['cost_usd']}")

                # Latency distribution chart
                if latency["count"] > 0:
                    chart_data = pd.DataFrame({
                        "Percentile": ["p50", "p75", "p90", "p95", "p99"],
                        "Latency (ms)": [
                            latency["p50"],
                            latency.get("p75", latency["p50"] * 1.3),
                            latency.get("p90", latency["p95"] * 0.9),
                            latency["p95"],
                            latency["p99"],
                        ]
                    })
                    st.bar_chart(chart_data.set_index("Percentile"))

st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub](https://github.com/gosidehustlesisi/sierra-genai-engineering)")
