# AI-Ready MLOps Framework
## Reusable ML Infrastructure Template

**🎯 What This Is:** A production-ready MLOps infrastructure template demonstrating model registry, drift detection, A/B testing, and auto-retraining patterns. This is **reusable infrastructure**, not a standalone analysis project — plug in your own data and models.

**💼 Use Case:** Production ML systems need operational guardrails. This template provides the scaffolding: model versioning, drift detection, A/B test routing, and rollback capability.

**🔧 Technical Stack:**
- **Data Sources (plug your own):**
  - U.S. Census ACS API — state-level demographics
  - BLS Public Data API — employment time series
  - arXiv API — CS paper abstracts
- **Tools:** Python, scikit-learn, SQLite, scipy, Streamlit
- **Pipeline:** Train → Register → Monitor → Detect Drift → Retrain → A/B Test → Promote

**📊 What's Included:**
- SQLite model registry with versioning and lineage tracking
- Statistical drift detection (KS test + PSI) with configurable thresholds
- A/B testing router with traffic splitting
- Auto-retraining trigger with regression gate
- Streamlit monitoring dashboard

**🏛️ Production Parallels:** Stripe (fraud model versioning), Netflix (A/B testing), Uber (drift detection)

---

## 📈 System Architecture

![Model Registry Version Timeline](figures/01_registry_versions.png)

*Figure 1: Model registry tracking accuracy and latency across staging and production versions*

---

## 🔍 Drift Detection Engine

![Drift Detection Dashboard](figures/02_drift_detection.png)

*Figure 2: Population Stability Index (PSI) and Kolmogorov-Smirnov tests across features, with 7-day alert timeline*

- **PSI < 0.1:** Stable (green)
- **PSI 0.1–0.25:** Warning (yellow) 
- **PSI > 0.25:** Critical (red) — triggers auto-retraining

---

## 🧪 A/B Testing Router

![A/B Testing](figures/03_ab_testing.png)

*Figure 3: Gradual traffic rollout with metric comparison between control and candidate models*

---

## 📡 Monitoring Dashboard

![Monitoring Overview](figures/04_monitoring_overview.png)

*Figure 4: System health gauges, request volume, error rates, latency percentiles, and model promotion pipeline*

---

## 🚀 How to Use

```bash
pip install -r requirements.txt
# Plug your own data and model
python -c "from src.retrain_pipeline import main; main()"
python src/drift_detector.py
streamlit run dashboard.py
```

**🖥️ Live Demo:** [Streamlit Dashboard](https://sierra-mlops-dashboard.streamlit.app) *(deploy your own with `streamlit run dashboard.py`)*

---

## 📂 Repository Structure

```
├── src/
│   ├── model_registry.py      # SQLite-backed MLflow-style registry
│   ├── drift_detector.py      # PSI + KS statistical drift detection
│   ├── ab_test.py            # A/B test routing framework
│   ├── retrain_pipeline.py   # Auto-retraining with regression gates
│   ├── monitor.py            # Health monitoring & alerting
│   └── deploy.py             # Model deployment utilities
├── notebooks/
│   └── 01_model_registry_demo.ipynb  # Interactive registry walkthrough
├── dashboard.py               # Streamlit monitoring UI
├── figures/                   # Generated visualizations
└── data/                      # SQLite registry + drift logs
```

**📄 License:** MIT

**About the Author:** Sierra Napier, MPA/MPH — AI Architect & Data Science Leader.
