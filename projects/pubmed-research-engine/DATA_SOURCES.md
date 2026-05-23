# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| NCBI E-utilities (PubMed) | Government API | Biomedical literature search and retrieval | https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ |

## Data Provenance

- 500 biomedical abstracts fetched live from PubMed E-utilities API on 2026-05-21
- Query: "machine learning" (sorted by date, most recent first)
- API: esearch.fcgi (ID lookup) + efetch.fcgi (XML metadata retrieval)
- No API key required for basic access; rate limit: ~3 requests/second
- Fields extracted: PMID, ArticleTitle, AbstractText, Journal/Title, PubDate/Year, AuthorList, MeshHeadingList
- XML parsed with Python ElementTree; structured into flat CSV

## Data Files

| File | Description | Size |
|------|-------------|------|
| pubmed_abstracts.csv | 500 abstracts with 10 columns | ~1.2 MB |
| pubmed_metadata.json | Fetch metadata with journal and year breakdown | ~5 KB |

## Refresh Strategy

- Re-run `python src/fetch_pubmed.py` to fetch updated abstracts
- PubMed updates daily; machine learning in biomedicine is a high-growth area
- For systematic reviews, register for NCBI API key to increase rate limits
- Alternative queries: "deep learning", "natural language processing", "artificial intelligence"
