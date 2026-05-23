# Data Sources

## Primary Data Sources

| Source | Type | Description | URL |
|--------|------|-------------|-----|
| CourtListener API | Legal Research API | Free case law database with full-text search | https://www.courtlistener.com/api/ |
| Supreme Court Database (SCDB) | Academic Database | Structured SCOTUS decision data (1937–present) | http://scdb.wustl.edu/ |
| Justia Supreme Court | Public Legal | Opinion text and case summaries | https://supreme.justia.com/ |

## Data Provenance

- CourtListener API: Free tier available (no key for basic access)
- SCDB: Academic database from Washington University in St. Louis
- Opinion text extracted from public domain Supreme Court decisions

## Data Files

No local CSV — data fetched on-demand via API or downloaded from public sources.

## Refresh Strategy

- SCOTUS releases opinions during term (October–June)
- CourtListener updates within hours of new decisions
- SCDB updates annually with coded decisions
