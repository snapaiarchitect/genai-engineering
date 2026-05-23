# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| ArXiv API | Real-time API | Scientific/financial abstracts | https://arxiv.org/help/api/index |
| PubMed E-utilities | Real-time API | Medical/health abstracts | https://www.ncbi.nlm.nih.gov/books/NBK25500/ |
| Wikipedia REST API | Real-time API | Legal/financial encyclopedia articles | https://www.mediawiki.org/wiki/API:Main_page |

## Data Provenance

- All documents fetched live via `src/download_documents.py`
- **ArXiv**: 455 scientific/financial abstracts (categories: cs.LG, cs.CL, q-fin)
- **PubMed**: 414 medical abstracts (search: machine learning, natural language processing, drug discovery, epidemiology)
- **Wikipedia**: 99 legal/financial articles (categories: Law, Finance, Economics)
- Total: 968 real documents with titles, abstracts, and source IDs
- `demo_100.csv` contains real arXiv IDs, PubMed PMIDs, and paper titles

## Data Files

| File | Description | Size |
|------|-------------|------|
| demo_100.csv | 100-row sample with real paper IDs and abstracts | 150 KB |
| reports/logistic_regression_metrics.json | Classification metrics | 308 B |
| reports/random_forest_metrics.json | Classification metrics | 274 B |

## Refresh Strategy

- Re-run `python src/download_documents.py` to fetch fresh corpus
- All APIs are free for research use
- Corpus changes as new papers are published
