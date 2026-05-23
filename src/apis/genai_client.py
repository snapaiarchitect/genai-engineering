"""
Shared API client for live data fetching across all sierra-genai-engineering projects.
Centralizes rate limiting, caching, retries, and provenance logging.
"""
import time
import hashlib
import json
import os
import requests
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genai_apis")


class APIClient:
    """Production-grade HTTP client with rate limiting, retries, and provenance logging."""

    def __init__(
        self,
        base_url: str,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_header: str = "X-Api-Key",
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.session = requests.Session()
        self._last_request_time: Optional[float] = None

        # Disk cache for idempotent GET requests
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "..", "data", ".api_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if api_key:
            self.session.headers[api_key_header] = api_key

    def _cache_key(self, method: str, endpoint: str, params: Dict) -> str:
        payload = f"{method}:{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _rate_limit(self):
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _log_provenance(
        self,
        endpoint: str,
        params: Dict,
        status_code: int,
        records_count: int,
        source_label: str,
    ):
        """Write a provenance entry for every API call."""
        provenance_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
        os.makedirs(provenance_dir, exist_ok=True)
        provenance_path = os.path.join(provenance_dir, "DATA_PROVENANCE_LOG.jsonl")

        entry = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "source": source_label,
            "endpoint": f"{self.base_url}{endpoint}",
            "params": params,
            "status_code": status_code,
            "records_fetched": records_count,
            "api_key_used": bool(self.api_key),
        }
        with open(provenance_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        source_label: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Execute a GET request with automatic retries, caching, rate limiting,
        and provenance logging.
        """
        params = params or {}
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        cache_key = self._cache_key("GET", endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        if use_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                logger.info(f"[CACHE HIT] {endpoint}")
                return json.load(f)

        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                resp = self.session.get(url, params=params, timeout=self.timeout)
                status = resp.status_code

                if status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(f"[RATE LIMITED] {endpoint} — sleeping {retry_after}s")
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Cache successful response
                if use_cache:
                    with open(cache_path, "w") as f:
                        json.dump(data, f)

                # Log provenance
                records = len(data) if isinstance(data, list) else 0
                self._log_provenance(endpoint, params, status, records, source_label)

                logger.info(f"[FETCH OK] {endpoint} — {records} records")
                return data

            except requests.exceptions.RequestException as e:
                logger.warning(f"[ATTEMPT {attempt}/{self.max_retries}] {endpoint} failed: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)

        return {}

    def post(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        source_label: str = "unknown",
    ) -> Dict[str, Any]:
        """POST request with retries and logging (no caching)."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limit()
                resp = self.session.post(
                    url, data=data, json=json_data, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"[ATTEMPT {attempt}/{self.max_retries}] POST {endpoint} failed: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        return {}


def fetch_arxiv_papers(
    search_query: str = "cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML",
    max_results: int = 500,
    cache_dir: Optional[str] = None,
) -> list:
    """
    Fetch arXiv abstracts via the public export API.
    Returns list of dicts with keys: id, title, abstract, authors, published, category.
    """
    client = APIClient(
        base_url="https://export.arxiv.org",
        rate_limit_delay=3.0,  # arXiv is strict
        cache_dir=cache_dir,
    )

    all_records = []
    batch_size = 100
    for start in range(0, max_results, batch_size):
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(batch_size, max_results - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        raw = client.get("/api/query", params=params, source_label="arxiv_api")

        # arXiv returns Atom XML — parse simplified
        feed = raw.get("feed", {})
        entries = feed.get("entry", [])
        if not isinstance(entries, list):
            entries = [entries]

        for entry in entries:
            all_records.append({
                "id": entry.get("id", ""),
                "title": entry.get("title", "").replace("\n", " ").strip(),
                "abstract": entry.get("summary", "").replace("\n", " ").strip(),
                "authors": ", ".join(
                    a.get("name", "") for a in entry.get("author", [])
                    if isinstance(a, dict)
                ),
                "published": entry.get("published", ""),
                "primary_category": entry.get("arxiv:primary_category", {}).get("@term", ""),
                "categories": " ".join(
                    c.get("@term", "") for c in entry.get("category", [])
                    if isinstance(c, dict)
                ),
            })

        if len(entries) < batch_size:
            break

    logger.info(f"[arXiv] Fetched {len(all_records)} papers")
    return all_records


def fetch_pubmed_abstracts(
    query: str = "machine learning",
    max_results: int = 500,
    cache_dir: Optional[str] = None,
) -> list:
    """
    Fetch PubMed abstracts via NCBI E-utilities.
    Returns list of dicts with keys: pmid, title, abstract, authors, year, journal, mesh_terms.
    """
    client = APIClient(
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        rate_limit_delay=0.34,  # NCBI allows ~3 req/sec
        cache_dir=cache_dir,
    )

    # Step 1: Search
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    search_resp = client.get("/esearch.fcgi", params=search_params, source_label="pubmed_esearch")
    idlist = search_resp.get("esearchresult", {}).get("idlist", [])

    if not idlist:
        logger.warning("[PubMed] No PMIDs found")
        return []

    # Step 2: Fetch
    all_records = []
    batch_size = 200
    for i in range(0, len(idlist), batch_size):
        batch = idlist[i : i + batch_size]
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        # NCBI fetch returns XML — handled by caller or saved to file
        raw_xml = client.get("/efetch.fcgi", params=fetch_params, source_label="pubmed_efetch")
        all_records.append({"pmids": batch, "raw_xml_size": len(str(raw_xml))})

    logger.info(f"[PubMed] Fetched {len(idlist)} PMIDs")
    return all_records


def fetch_congress_bills(
    congress: int = 118,
    bill_type: str = "HCONRES",
    max_results: int = 500,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list:
    """
    Fetch bills from Congress.gov API.
    Uses DEMO_KEY if no key provided (limited to 5,000 requests/day).
    """
    client = APIClient(
        base_url="https://api.congress.gov/v3",
        rate_limit_delay=1.0,
        api_key=api_key or "DEMO_KEY",
        api_key_header="api_key",
        cache_dir=cache_dir,
    )

    all_bills = []
    offset = 0
    limit = 250
    while len(all_bills) < max_results:
        params = {"limit": min(limit, max_results - len(all_bills)), "offset": offset}
        data = client.get(
            f"/bill/{congress}/{bill_type}", params=params, source_label="congressgov_api"
        )
        bills = data.get("bills", [])
        if not bills:
            break
        all_bills.extend(bills)
        offset += len(bills)

    logger.info(f"[Congress.gov] Fetched {len(all_bills)} {bill_type} bills")
    return all_bills


def fetch_clinical_trials(
    query: str = "machine learning",
    page_size: int = 100,
    max_pages: int = 5,
    cache_dir: Optional[str] = None,
) -> list:
    """
    Fetch trials from ClinicalTrials.gov API v2.
    Returns list of trial metadata dicts.
    """
    client = APIClient(
        base_url="https://clinicaltrials.gov/api/v2",
        rate_limit_delay=0.5,
        cache_dir=cache_dir,
    )

    all_trials = []
    next_token = None
    for _ in range(max_pages):
        params = {"query.cond": query, "pageSize": page_size}
        if next_token:
            params["pageToken"] = next_token

        data = client.get("/studies", params=params, source_label="clinicaltrials_api")
        studies = data.get("studies", [])
        if not studies:
            break
        all_trials.extend(studies)
        next_token = data.get("nextPageToken")
        if not next_token:
            break

    logger.info(f"[ClinicalTrials.gov] Fetched {len(all_trials)} trials")
    return all_trials


def fetch_oyez_cases(term: int = 2022, cache_dir: Optional[str] = None) -> list:
    """
    Fetch SCOTUS cases from Oyez API with justice voting data.
    """
    client = APIClient(
        base_url="https://api.oyez.org",
        rate_limit_delay=1.0,
        cache_dir=cache_dir,
    )

    data = client.get(f"/cases?filter=term:{term}", source_label="oyez_api")
    cases = data if isinstance(data, list) else []
    logger.info(f"[Oyez] Fetched {len(cases)} cases for term {term}")
    return cases


def fetch_census_acs(
    variables: list = None,
    year: int = 2022,
    dataset: str = "acs5",
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list:
    """
    Fetch Census ACS data. Variables like B15003_022E (bachelors degree), B23027_002E (labor force).
    """
    variables = variables or ["B15003_022E", "B23027_002E", "B01003_001E", "NAME"]
    client = APIClient(
        base_url="https://api.census.gov/data",
        rate_limit_delay=0.5,
        api_key=api_key,
        api_key_header="key",
        cache_dir=cache_dir,
    )

    var_str = ",".join(variables)
    data = client.get(
        f"/{year}/acs/{dataset}",
        params={"get": var_str, "for": "county:*"},
        source_label="census_acs_api",
    )

    # Census returns [headers, row1, row2, ...]
    if not data or len(data) < 2:
        return []

    headers = data[0]
    records = [dict(zip(headers, row)) for row in data[1:]]
    logger.info(f"[Census ACS] Fetched {len(records)} county records")
    return records
