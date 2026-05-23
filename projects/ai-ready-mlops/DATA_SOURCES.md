# Data Sources

## Primary Data Sources

This is a **reusable MLOps infrastructure template** — it does not ship with fixed data. Instead, it provides plug points for your own datasets.

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| (Plug your own) | Any | The framework accepts any tabular or text data | — |

## Example Data Sources You Can Plug In

| Source | Type | Use Case | URL |
|--------|------|----------|-----|
| U.S. Census ACS API | Real-time API | Demographic drift monitoring | https://api.census.gov/ |
| BLS Public Data API | Real-time API | Employment trend drift | https://www.bls.gov/developers/ |
| arXiv API | Real-time API | NLP model drift on CS abstracts | https://export.arxiv.org/api/query |

## Data Provenance

- No fixed dataset included — this is infrastructure code
- `src/retrain_pipeline.py` generates synthetic drift metrics for demonstration only
- Replace with your own data loader to productionize

## Refresh Strategy

- Connect your own data source to `src/retrain_pipeline.py`
- Re-run pipeline on schedule (cron / Airflow / CI) for continuous monitoring
