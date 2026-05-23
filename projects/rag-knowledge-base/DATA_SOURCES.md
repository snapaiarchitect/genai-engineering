# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| arXiv API | Real-time API | 2,646 scientific abstracts with metadata | https://export.arxiv.org/api/query |

## Data Provenance

- All abstracts fetched live from arXiv.org API via `src/download_corpus.py`
- 8 topic queries: machine learning, NLP, computer vision, data science, deep learning, reinforcement learning, neural networks, large language models
- Categories: cs.LG, cs.CL, cs.AI, cs.CV, stat.ML
- Each abstract includes: title, authors, summary, categories, published date, arXiv URL
- Fetched 2026 (recent) — all records have year=2026 in corpus_stats.json

## Data Files

| File | Description | Size |
|------|-------------|------|
| corpus_stats.json | Aggregate metadata: 2,646 docs, category counts, length stats | ~2 KB |
| query_benchmarks.json | 10 benchmark queries with retrieved paper titles | ~5 KB |

## Refresh Strategy

- Re-run `python src/download_corpus.py` to fetch fresh abstracts
- arXiv API is free, no rate limit for reasonable usage
- Corpus evolves as new preprints are published
