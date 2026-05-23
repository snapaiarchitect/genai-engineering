# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| ClinicalTrials.gov API v2 | Government API | Live clinical trial registry data | https://clinicaltrials.gov/api/v2/studies |

## Data Provenance

- All 500 trials fetched live from ClinicalTrials.gov API on 2026-05-21
- Query: `condition=cancer`, `filter.overallStatus=RECRUITING`, `pageSize=100`
- 5 paginated requests with 0.5s delay between pages (rate limit compliance)
- Each record includes: NCT ID, title, condition, phase, sponsor, enrollment, locations, intervention type, study type, start/completion dates

## Data Files

| File | Description | Size |
|------|-------------|------|
| clinical_trials.csv | 500 active cancer trial records | ~144 KB |
| trial_metadata.json | Fetch metadata: date, API version, record counts | ~31 KB |

## Refresh Strategy

- Re-run `python src/fetch_trials.py` to fetch fresh trial data
- ClinicalTrials.gov updates in real-time as sponsors register new trials
- API is free with no key required for reasonable usage
- Full documentation: https://clinicaltrials.gov/data-api/api
