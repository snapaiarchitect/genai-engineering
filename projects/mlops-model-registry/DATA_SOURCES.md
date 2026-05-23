# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| U.S. Census ACS 5-Year API | Government API | American Community Survey demographic/economic data | https://api.census.gov/data/2022/acs/acs5 |

## Data Provenance

- 3,222 U.S. counties fetched live from Census ACS API on 2026-05-21
- API key: Set via `CENSUS_API_KEY` environment variable. Get yours at https://api.census.gov/data/key_signup.html
- Year: 2022 (most recent 5-year estimates available)
- Variables: B19013_001E (median income), B01003_001E (population), B15003_022E (bachelor's), B23027_002E (labor force), B25064_001E (median rent), B08303_001E (commute time), B25003_002E/003E (housing tenure)
- All numeric fields validated — zero/negative income values filtered

## Data Files

| File | Description | Size |
|------|-------------|------|
| acs_county_data.csv | 3,222 county records with 9 columns | ~290 KB |
| acs_metadata.json | Fetch metadata with variable definitions | ~3 KB |

## Refresh Strategy

- Re-run `python src/fetch_acs_data.py` to fetch updated ACS estimates
- Census releases new 5-year estimates annually (latest: 2023, available late 2024)
- API key required: register at https://api.census.gov/data/key_signup.html
