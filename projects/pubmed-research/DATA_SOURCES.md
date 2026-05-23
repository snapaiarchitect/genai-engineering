# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| PubMed E-utilities | Government API | NCBI search and retrieval for MEDLINE citations | https://www.ncbi.nlm.nih.gov/home/develop/api/ |
| PubMed Central (PMC) | Government Repository | Full-text open access articles | https://www.ncbi.nlm.nih.gov/pmc/ |
| MeSH Database | Government Ontology | Medical Subject Headings vocabulary | https://meshb.nlm.nih.gov/ |

## Data Provenance

- Fetched via NCBI E-utilities (Esearch, Esummary, Efetch)
- API key recommended for high-volume access (>3 requests/second)
- Coverage: 35+ million MEDLINE citations, 1946–present

## Data Files

No local CSV — data fetched on-demand via API and processed in notebooks.

## Refresh Strategy

- PubMed updates daily with new citations
- NCBI E-utilities available 24/7
- API key request: https://www.ncbi.nlm.nih.gov/account/
