# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| arXiv API | Academic API | Open access preprint papers in physics, math, CS, biology | http://export.arxiv.org/api/query |
| arXiv OAI | Academic API | Bulk metadata harvest via OAI-PMH | http://export.arxiv.org/oai2 |

## Data Provenance

- Fetched via arXiv public API (no key required, rate-limited)
- Categories queried: cs.LG, cs.AI, cs.CL
- Fields extracted: title, abstract, primary_category, authors, published date
- Respects arXiv rate limits (3-second delay between requests)

## Data Files

| File | Description | Size |
|------|-------------|------|
| arxiv_papers.csv | Fetched paper metadata and abstracts | ~500 rows |

## Refresh Strategy

- arXiv updates daily with new submissions
- Re-fetch via API for latest papers
- No API key required — public academic resource
