# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| Congress.gov API v3 | Government API | Official U.S. legislative data | https://api.congress.gov/ |

## Data Provenance

- 496 House Concurrent Resolutions (HCONRES) fetched live from Congress.gov API on 2026-05-21
- API key: DEMO_KEY (development rate limits apply)
- Congress: 118th (2023–2025)
- Bill types in sample: HCONRES only — due to DEMO_KEY rate limiting and API sorting behavior
- Each record includes: congress, number, type, originChamber, title, latestAction date/text, updateDate

## Data Files

| File | Description | Size |
|------|-------------|------|
| congress_bills.csv | 496 legislative records | ~150 KB |
| bill_metadata.json | Fetch metadata: types, chambers, counts | ~2 KB |

## Refresh Strategy

- Re-run `python src/fetch_bills.py` to fetch fresh records
- Congress.gov API updates in real-time as legislative actions occur
- Production use requires registration at https://api.congress.gov/ for dedicated API key
- Rate limit: DEMO_KEY allows limited requests per hour
