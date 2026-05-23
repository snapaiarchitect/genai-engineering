# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| Oyez API | Legal Research API | Supreme Court case data with justice voting records | https://api.oyez.org/ |

## Data Provenance

- 59 decided SCOTUS cases from the 2022 term fetched live from Oyez API on 2026-05-21
- Each case includes: case name, docket number, question presented, timeline (granted/argued/decided), decision data (majority/minority votes, winning party, decision type), and individual justice votes with opinion types
- Justices observed: John G. Roberts Jr., Clarence Thomas, Samuel A. Alito Jr., Sonia Sotomayor, Elena Kagan, Neil Gorsuch, Brett M. Kavanaugh, Amy Coney Barrett, Ketanji Brown Jackson
- Term: October 2022 – June 2023

## Data Files

| File | Description | Size |
|------|-------------|------|
| scotus_cases.csv | 59 cases with 30+ columns including justice votes | ~45 KB |
| case_metadata.json | Fetch metadata with justice list and date range | ~2 KB |

## Refresh Strategy

- Re-run `python src/fetch_cases.py` to fetch updated cases
- Oyez updates as new terms conclude (typically July each year)
- No API key required for basic access
- Rate limit: ~0.3s between requests (courteous scraping)

## Differentiation from `scotus-opinions`

| Aspect | `scotus-opinions` | `nlp-supreme-court-opinion-analysis` |
|--------|-------------------|--------------------------------------|
| Data source | CourtListener API + SCDB | Oyez API |
| Analytical lens | Opinion text NLP (sentiment, topic modeling, citations) | Justice voting behavior (alignment, margins, timelines) |
| Primary output | Text embeddings, topic clusters, citation networks | Voting matrices, outcome prediction, doctrinal blocs |
| Figure count | 5 text-focused | 6 behavior-focused |
