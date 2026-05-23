#!/usr/bin/env python3
"""
Fetch SCOTUS opinion data using live APIs with fallback to preserved data.

API Strategy:
  1. CourtListener (https://www.courtlistener.com/api/)
     - REST API v3 endpoints: /api/rest/v3/opinions/, /api/rest/v3/clusters/
     - Requires API token (set COURTLISTENER_API_TOKEN env var)
  2. Oyez (https://api.oyez.org/)
     - Case detail: /cases/{term}/{docket_number}
     - No auth required, but term list endpoints have a known caching bug
     - We use a curated list of landmark cases with direct URLs
  3. Preserved landmark data (original synthetic data)
     - Ultimate fallback when both APIs fail

Output Schema (same as original):
  - case_name, citation, term, chief_justice, majority_author
  - word_count, topic, disposition, votes_for, votes_against
  - opinion_text, source
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import requests
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
CACHE_DIR = Path.home() / '.cache' / 'scotus_opinions'
CACHE_TTL_DAYS = 7
REQUEST_DELAY = 1.0  # seconds between API calls (rate limiting)
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
USER_AGENT = 'SCOTUS-Data-Fetcher/1.0 (Research Project; https://github.com/gosidehustlesisi/sierra-genai-engineering)'

# CourtListener
COURTLISTENER_BASE = 'https://www.courtlistener.com/api/rest/v3'
COURTLISTENER_TOKEN = os.environ.get('COURTLISTENER_API_TOKEN', '')

# Oyez
OYEZ_BASE = 'https://api.oyez.org'

# ── Curated Landmark Cases for Oyez Direct Fetch ───────────────────────────
# Format: (case_name, term, docket_number, expected_citation)
LANDMARK_CASES = [
    ("Marbury v. Madison", "1803", "1", "5 U.S. 137"),
    ("McCulloch v. Maryland", "1819", "35", "17 U.S. 316"),
    ("Gibbons v. Ogden", "1824", "21", "22 U.S. 1"),
    ("Dred Scott v. Sandford", "1856", "446", "60 U.S. 393"),
    ("Munn v. Illinois", "1876", "94", "94 U.S. 113"),
    ("Plessy v. Ferguson", "1896", "210", "163 U.S. 537"),
    ("Schenck v. United States", "1918", "437", "249 U.S. 47"),
    ("Gitlow v. New York", "1923", "19", "268 U.S. 652"),
    ("Near v. Minnesota", "1929", "91", "283 U.S. 697"),
    ("West Coast Hotel v. Parrish", "1936", "306", "300 U.S. 379"),
    ("United States v. Darby", "1940", "121", "312 U.S. 100"),
    ("Wickard v. Filburn", "1942", "59", "317 U.S. 111"),
    ("Korematsu v. United States", "1944", "22", "323 U.S. 214"),
    ("Youngstown Sheet & Tube Co. v. Sawyer", "1952", "1", "343 U.S. 579"),
    ("Brown v. Board of Education", "1953", "347", "347 U.S. 483"),
    ("Bolling v. Sharpe", "1954", "8", "347 U.S. 497"),
    ("Roth v. United States", "1956", "582", "354 U.S. 476"),
    ("Mapp v. Ohio", "1960", "165", "367 U.S. 643"),
    ("Baker v. Carr", "1961", "6", "369 U.S. 186"),
    ("Engel v. Vitale", "1962", "155", "370 U.S. 421"),
    ("Gideon v. Wainwright", "1962", "155", "372 U.S. 335"),
    ("New York Times v. Sullivan", "1963", "39", "376 U.S. 254"),
    ("Heart of Atlanta Motel v. United States", "1964", "357", "379 U.S. 241"),
    ("Griswold v. Connecticut", "1965", "496", "381 U.S. 479"),
    ("Miranda v. Arizona", "1966", "759", "384 U.S. 436"),
    ("Katz v. United States", "1967", "348", "389 U.S. 347"),
    ("Brandenburg v. Ohio", "1968", "895", "395 U.S. 444"),
    ("Tinker v. Des Moines", "1968", "21", "393 U.S. 503"),
    ("New York Times Co. v. United States", "1970", "1873", "403 U.S. 713"),
    ("Roe v. Wade", "1971", "70-18", "410 U.S. 113"),
    ("United States v. Nixon", "1973", "73-1766", "418 U.S. 683"),
    ("Buckley v. Valeo", "1975", "76-436", "424 U.S. 1"),
    ("Regents v. Bakke", "1977", "76-811", "438 U.S. 265"),
    ("Roberts v. United States Jaycees", "1983", "83-724", "468 U.S. 609"),
    ("Texas v. Johnson", "1988", "88-155", "491 U.S. 397"),
    ("Planned Parenthood v. Casey", "1991", "91-744", "505 U.S. 833"),
    ("United States v. Lopez", "1994", "93-1260", "514 U.S. 549"),
    ("City of Boerne v. Flores", "1996", "95-2074", "521 U.S. 507"),
    ("Bush v. Gore", "2000", "00-949", "531 U.S. 98"),
    ("Lawrence v. Texas", "2002", "02-102", "539 U.S. 558"),
    ("District of Columbia v. Heller", "2007", "07-290", "554 U.S. 570"),
    ("Citizens United v. FEC", "2008", "08-205", "558 U.S. 310"),
    ("National Federation of Independent Business v. Sebelius", "2011", "11-393", "567 U.S. 519"),
    ("United States v. Windsor", "2012", "12-307", "570 U.S. 744"),
    ("Obergefell v. Hodges", "2014", "14-556", "576 U.S. 644"),
]

# ── Cache Helpers ─────────────────────────────────────────────────────────

def _cache_key(prefix: str, identifier: str) -> str:
    """Generate a cache file name from a prefix and identifier."""
    h = hashlib.md5(identifier.encode()).hexdigest()[:12]
    return f"{prefix}_{h}.json"


def _cache_path(prefix: str, identifier: str) -> Path:
    """Return the full path for a cache file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / _cache_key(prefix, identifier)


def _load_cache(prefix: str, identifier: str) -> Optional[Any]:
    """Load cached data if it exists and is not expired."""
    path = _cache_path(prefix, identifier)
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=CACHE_TTL_DAYS):
            logger.info("Cache expired for %s", identifier)
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache load failed for %s: %s", identifier, e)
        return None


def _save_cache(prefix: str, identifier: str, data: Any) -> None:
    """Save data to cache."""
    path = _cache_path(prefix, identifier)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("Cache save failed for %s: %s", identifier, e)


# ── Request Helpers ─────────────────────────────────────────────────────────

def _request_with_retry(url: str, headers: Optional[Dict] = None,
                        params: Optional[Dict] = None,
                        timeout: int = 30) -> Optional[requests.Response]:
    """Make an HTTP GET request with retry logic and rate limiting."""
    merged_headers = {'User-Agent': USER_AGENT}
    if headers:
        merged_headers.update(headers)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=merged_headers, params=params,
                                timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (429, 502, 503, 504):
                wait = BACKOFF_FACTOR ** attempt + random.uniform(0, 1)
                logger.warning("Rate limit/server error %s for %s, retrying in %.1fs (attempt %d/%d)",
                               resp.status_code, url, wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
            else:
                logger.error("HTTP %s for %s", resp.status_code, url)
                return resp  # Return to let caller handle
        except requests.exceptions.RequestException as e:
            wait = BACKOFF_FACTOR ** attempt + random.uniform(0, 1)
            logger.warning("Request error for %s: %s, retrying in %.1fs (attempt %d/%d)",
                           url, e, wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)

    logger.error("Failed to fetch %s after %d retries", url, MAX_RETRIES)
    return None


def _fetch_json(url: str, headers: Optional[Dict] = None,
                params: Optional[Dict] = None,
                cache_prefix: str = "api") -> Optional[Dict]:
    """Fetch JSON from URL with caching and rate limiting."""
    cache_id = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
    cached = _load_cache(cache_prefix, cache_id)
    if cached is not None:
        logger.debug("Cache hit for %s", url)
        return cached

    resp = _request_with_retry(url, headers=headers, params=params)
    if resp is None:
        return None

    if resp.status_code != 200:
        logger.error("HTTP %s for %s: %s", resp.status_code, url, resp.text[:200])
        return None

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        logger.error("JSON decode error for %s: %s", url, e)
        return None

    _save_cache(cache_prefix, cache_id, data)
    time.sleep(REQUEST_DELAY)
    return data


# ── CourtListener API ─────────────────────────────────────────────────────

def _courtlistener_headers() -> Dict[str, str]:
    """Build headers for CourtListener API."""
    h = {'User-Agent': USER_AGENT}
    if COURTLISTENER_TOKEN:
        h['Authorization'] = f'Token {COURTLISTENER_TOKEN}'
    return h


def fetch_courtlistener_opinions(max_cases: int = 20) -> List[Dict]:
    """Fetch SCOTUS opinions from CourtListener API."""
    if not COURTLISTENER_TOKEN:
        logger.info("No COURTLISTENER_API_TOKEN set, skipping CourtListener")
        return []

    url = f"{COURTLISTENER_BASE}/opinions/"
    params = {
        'court': 'scotus',
        'page_size': min(max_cases, 50),
        'ordering': '-date_created',
    }

    logger.info("Fetching from CourtListener: %s", url)
    data = _fetch_json(url, headers=_courtlistener_headers(), params=params,
                       cache_prefix="courtlistener")

    if data is None:
        return []

    results = data.get('results', [])
    if not results:
        logger.warning("CourtListener returned empty results")
        return []

    cases = []
    for item in results[:max_cases]:
        case = _parse_courtlistener_item(item)
        if case:
            cases.append(case)

    logger.info("Fetched %d cases from CourtListener", len(cases))
    return cases


def _parse_courtlistener_item(item: Dict) -> Optional[Dict]:
    """Parse a CourtListener API response item into our schema."""
    try:
        case_name = item.get('case_name') or item.get('case_name_full') or 'Unknown'
        date_str = item.get('date_filed') or item.get('date_created', '')
        term = ''
        if date_str:
            try:
                term = str(datetime.strptime(date_str[:10], '%Y-%m-%d').year)
            except ValueError:
                pass

        # Try to extract citation
        citation = ''
        citations = item.get('citations', [])
        if citations:
            citation = citations[0].get('text', '')
        if not citation:
            citation = item.get('citation', '')

        # Get opinion text
        text = item.get('plain_text', '') or item.get('html', '') or item.get('text', '')
        text = _strip_html(text)

        # Word count
        word_count = len(text.split()) if text else 0

        # Topic inference
        topic = _derive_topic(case_name, text)

        # Votes - CourtListener doesn't always have vote counts
        votes_for = item.get('votes_for', 0) or 0
        votes_against = item.get('votes_against', 0) or 0

        # Disposition
        disposition = item.get('status', '') or item.get('decision_direction', '') or 'Unknown'

        # Justices info - often not available in opinions endpoint
        majority_author = ''
        chief_justice = ''

        return {
            'case_name': case_name,
            'citation': citation,
            'term': term,
            'chief_justice': chief_justice,
            'majority_author': majority_author,
            'word_count': word_count,
            'topic': topic,
            'disposition': disposition,
            'votes_for': votes_for,
            'votes_against': votes_against,
            'opinion_text': text[:10000],  # Limit text length
            'source': 'courtlistener',
        }
    except Exception as e:
        logger.error("Error parsing CourtListener item: %s", e)
        return None


# ── Oyez API ──────────────────────────────────────────────────────────────

def fetch_oyez_case(term: str, docket_number: str, expected_name: str = "") -> Optional[Dict]:
    """Fetch a single SCOTUS case from Oyez API with validation."""
    url = f"{OYEZ_BASE}/cases/{term}/{docket_number}"
    logger.info("Fetching Oyez case: %s", url)
    data = _fetch_json(url, cache_prefix="oyez")
    if data is None:
        return None
    return _parse_oyez_case(data, term, docket_number, expected_name)


def _parse_oyez_case(data: Dict, term: str, docket_number: str, expected_name: str = "") -> Optional[Dict]:
    """Parse an Oyez case detail response into our schema.
    
    Validates that the returned data matches the requested case.
    """
    try:
        # Handle both object and array responses
        case_data = None
        
        if isinstance(data, list):
            if not data:
                return None
            # Oyez sometimes returns a list - find matching case by term and docket
            for item in data:
                if isinstance(item, dict):
                    item_term = str(item.get('term', ''))
                    item_docket = str(item.get('docket_number', ''))
                    if item_term == term and item_docket == docket_number:
                        case_data = item
                        break
            # If no match found, try matching by expected name
            if case_data is None and expected_name:
                for item in data:
                    if isinstance(item, dict):
                        item_name = item.get('name', '')
                        if expected_name.lower() in item_name.lower() or item_name.lower() in expected_name.lower():
                            case_data = item
                            break
            # Last resort: take first item if it's a reasonable case
            if case_data is None:
                case_data = data[0]
        elif isinstance(data, dict):
            case_data = data
        else:
            logger.warning("Unexpected Oyez response type: %s", type(data))
            return None

        # Validate that this is the case we requested
        case_name = case_data.get('name', '')
        returned_term = str(case_data.get('term', ''))
        returned_docket = str(case_data.get('docket_number', ''))
        
        # Check if the returned case matches what we requested
        matches_request = (
            (returned_term == term and returned_docket == docket_number) or
            (expected_name and expected_name.lower() in case_name.lower())
        )
        
        # If Oyez returned a canned list (same cases for all endpoints), 
        # the term/docket won't match. Detect this common bug.
        if not matches_request and expected_name:
            logger.warning(
                "Oyez returned wrong case for %s/%s: got '%s' (term=%s, docket=%s)",
                term, docket_number, case_name, returned_term, returned_docket
            )
            # Try once more with a direct lookup if we have a case ID
            case_id = case_data.get('ID')
            if case_id:
                detail_url = f"{OYEZ_BASE}/cases/{case_id}"
                logger.info("Trying direct case ID lookup: %s", detail_url)
                detail_data = _fetch_json(detail_url, cache_prefix="oyez")
                if detail_data and isinstance(detail_data, dict):
                    return _parse_oyez_case(detail_data, term, docket_number, expected_name)
            return None

        # Build text from available fields
        text_parts = []
        for field in ['description', 'facts_of_the_case', 'question', 'conclusion']:
            val = case_data.get(field, '')
            if val and isinstance(val, str):
                text_parts.append(f"{field.replace('_', ' ').title()}: {val}")

        # Try to get opinion text from decisions
        decisions = case_data.get('decisions', [])
        majority_text = ''
        votes_for = 0
        votes_against = 0
        majority_author = ''

        for decision in decisions:
            if isinstance(decision, dict):
                # Try to find majority opinion
                for opinion in decision.get('opinions', []):
                    if isinstance(opinion, dict):
                        opinion_type = opinion.get('type', '').lower()
                        if 'majority' in opinion_type or opinion_type == 'majority':
                            text = opinion.get('text', '') or ''
                            if text:
                                majority_text = text
                            author = opinion.get('author', {})
                            if isinstance(author, dict):
                                majority_author = author.get('name', '')

                # Vote counts
                vote = decision.get('vote', {})
                if isinstance(vote, dict):
                    majority = vote.get('majority', 0)
                    minority = vote.get('minority', 0)
                    if majority or minority:
                        votes_for = majority
                        votes_against = minority

        if majority_text:
            text_parts.append(f"Majority Opinion: {majority_text}")

        text = '\n\n'.join(text_parts)
        text = _strip_html(text)
        word_count = len(text.split()) if text else 0

        # Get chief justice from heard_by or decided_by
        chief_justice = ''
        for field in ['decided_by', 'heard_by']:
            justices = case_data.get(field, [])
            if isinstance(justices, list) and justices:
                for justice in justices:
                    if isinstance(justice, dict) and justice.get('role') == 'chief justice':
                        chief_justice = justice.get('name', '')
                        break
            if chief_justice:
                break

        # Topic inference
        topic = _derive_topic(case_name, text)

        # Disposition
        disposition = 'Unknown'
        conclusion = case_data.get('conclusion', '')
        if conclusion and isinstance(conclusion, str):
            disposition = _extract_disposition(conclusion)

        # Citation handling - Oyez returns object or string
        citation = ''
        raw_citation = case_data.get('citation', '')
        if isinstance(raw_citation, dict):
            # Oyez citation object: {volume, page, year, href}
            vol = raw_citation.get('volume', '')
            pg = raw_citation.get('page', '')
            yr = raw_citation.get('year', '')
            if vol and pg:
                citation = f"{vol} U.S. {pg}"
            elif vol:
                citation = f"{vol} U.S."
        elif isinstance(raw_citation, str) and raw_citation:
            citation = raw_citation
        
        # Fallback: extract from justia_url
        if not citation and case_data.get('justia_url'):
            citation = _extract_citation_from_justia(case_data['justia_url'])

        return {
            'case_name': case_name,
            'citation': citation,
            'term': returned_term or term,
            'chief_justice': chief_justice,
            'majority_author': majority_author,
            'word_count': word_count,
            'topic': topic,
            'disposition': disposition,
            'votes_for': votes_for,
            'votes_against': votes_against,
            'opinion_text': text[:10000],
            'source': 'oyez',
        }
    except Exception as e:
        logger.error("Error parsing Oyez case %s/%s: %s", term, docket_number, e)
        return None


def fetch_oyez_landmark_cases(max_cases: int = 30) -> List[Dict]:
    """Fetch landmark SCOTUS cases from Oyez using curated list."""
    cases = []
    for case_name, term, docket, expected_citation in LANDMARK_CASES[:max_cases]:
        case = fetch_oyez_case(term, docket, expected_name=case_name)
        if case:
            # Ensure citation if API didn't provide one
            if not case.get('citation'):
                case['citation'] = expected_citation
            cases.append(case)
            logger.info("✓ Fetched: %s", case_name)
        else:
            logger.warning("✗ Failed: %s", case_name)

    logger.info("Fetched %d/%d landmark cases from Oyez", len(cases), min(max_cases, len(LANDMARK_CASES)))
    return cases


def _extract_citation_from_justia(url: str) -> str:
    """Try to extract citation from a Justia URL."""
    # URL format: https://supreme.justia.com/cases/federal/us/410/113/case.html
    match = re.search(r'/us/(\d+)/(\d+)/', url)
    if match:
        volume, page = match.groups()
        return f"{volume} U.S. {page}"
    return ''


# ── Text Processing ───────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _derive_topic(case_name: str, text: str) -> str:
    """Derive a topic category from case name and text."""
    name_lower = case_name.lower()
    text_lower = text.lower()
    combined = name_lower + ' ' + text_lower

    topic_keywords = {
        'First Amendment': ['first amendment', 'free speech', 'freedom of speech', 'press',
                           'religion', 'establishment', 'free exercise', 'assembly',
                           'petition', 'obscenity', 'libel', 'defamation', 'protest',
                           'symbolic speech', 'commercial speech'],
        'Fourth Amendment': ['fourth amendment', 'search', 'seizure', 'warrant',
                            'probable cause', 'reasonable suspicion', 'exclusionary rule',
                            'wiretap', 'surveillance', 'privacy', 'unreasonable search'],
        'Fifth Amendment': ['fifth amendment', 'due process', 'self-incrimination',
                           'double jeopardy', 'grand jury', 'eminent domain',
                           'miranda', 'confession', 'custodial interrogation'],
        'Sixth Amendment': ['sixth amendment', 'counsel', 'attorney', 'speedy trial',
                           'confrontation', 'compulsory process', 'jury trial',
                           'gideon', 'right to counsel'],
        'Fourteenth Amendment': ['fourteenth amendment', 'equal protection',
                                'due process clause', 'incorporation', 'substantive due process'],
        'Commerce Clause': ['commerce clause', 'interstate commerce', 'dormant commerce',
                           'necessary and proper', 'wagner act', 'labor relations',
                           'nlrb', 'stream of commerce'],
        'Civil Rights': ['civil rights', 'discrimination', 'segregation',
                        'desegregation', 'integration', 'affirmative action',
                        'voting rights', 'disability', 'title vii', 'equal employment'],
        'Abortion': ['abortion', 'roe', 'planned parenthood', 'casey', 'pro-life', 'pro-choice',
                    'trimester', 'undue burden', 'viability', 'dobbs'],
        'Marriage Equality': ['same-sex', 'gay marriage', 'marriage equality',
                             'domestic partnership', 'sexual orientation',
                             'obergefell', 'windsor', 'sodomy', 'lawrence'],
        'Gun Rights': ['second amendment', 'gun', 'firearm', 'handgun',
                      'concealed carry', 'heller', 'mcdonald', 'assault weapon'],
        'Executive Power': ['executive power', 'executive privilege', 'war powers',
                             'commander in chief', 'emergency power',
                             'youngstown', 'nixon', 'trump'],
        'Federalism': ['federalism', 'states rights', 'state sovereignty',
                      'commandeering', 'anti-commandeering', 'tenth amendment',
                      'new york v. united states', 'printz', 'lopez'],
        'Economic Regulation': ['economic', 'antitrust', 'monopoly', ' Sherman act',
                               'Clayton act', 'price fixing', 'merger',
                               'securities', 'banking', 'labor'],
        'Immigration': ['immigration', 'deportation', 'asylum', 'refugee',
                       'visa', 'citizenship', 'naturalization', 'border',
                       'dreamers', 'daca'],
        'National Security': ['national security', 'espionage', 'terrorism',
                             'guantanamo', 'enemy combatant', 'military commission',
                             'hamdan', 'hamdi', 'boumediene'],
        'Administrative Law': ['administrative', 'agency', 'chevron', 'rulemaking',
                              'arbitrary and capricious', ' APA ', 'regulation',
                              'delegation', 'non-delegation'],
        'Election Law': ['election', 'voting', 'redistricting', 'gerrymander',
                        'campaign finance', 'citizens united', 'bush v. gore',
                        'ballot', 'poll tax', 'literacy test', 'one person one vote'],
        'Property Rights': ['property', 'takings', 'eminent domain', 'zoning',
                           'land use', ' regulatory taking ', 'kelo'],
        'Criminal Procedure': ['criminal', 'death penalty', 'capital punishment',
                                'sentencing', 'habeas corpus', 'plea bargain',
                                'bail', 'probation', 'parole'],
    }

    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw in combined:
                return topic

    # Default topics based on broader patterns
    if any(w in combined for w in ['constitution', 'constitutional']):
        return 'Constitutional Structure'
    if any(w in combined for w in ['tort', 'negligence', 'liability']):
        return 'Torts'
    if any(w in combined for w in ['contract', 'breach']):
        return 'Contracts'
    if any(w in combined for w in ['patent', 'copyright', 'trademark', 'intellectual property']):
        return 'Intellectual Property'
    if any(w in combined for w in ['tax', 'revenue']):
        return 'Taxation'
    if any(w in combined for w in ['environment', 'epa', 'pollution', 'clean air', 'clean water']):
        return 'Environmental Law'
    if any(w in combined for w in ['health care', 'medicare', 'medicaid', 'affordable care act', 'aca']):
        return 'Health Care'

    return 'General Jurisdiction'


def _extract_disposition(conclusion: str) -> str:
    """Extract disposition from conclusion text."""
    conclusion_lower = conclusion.lower()
    if 'affirmed' in conclusion_lower:
        return 'Affirmed'
    elif 'reversed' in conclusion_lower:
        return 'Reversed'
    elif 'vacated' in conclusion_lower:
        return 'Vacated'
    elif 'remanded' in conclusion_lower:
        return 'Remanded'
    elif 'dismissed' in conclusion_lower:
        return 'Dismissed'
    elif 'granted' in conclusion_lower and 'petition' in conclusion_lower:
        return 'Petition Granted'
    elif 'denied' in conclusion_lower and 'petition' in conclusion_lower:
        return 'Petition Denied'
    return 'Decided'


# ── Preserved Landmark Data (Ultimate Fallback) ───────────────────────────

PRESERVED_LANDMARK_CASES = [
    {
        "case_name": "Marbury v. Madison",
        "citation": "5 U.S. 137",
        "term": "1803",
        "chief_justice": "John Marshall",
        "majority_author": "John Marshall",
        "word_count": 4200,
        "topic": "Judicial Review",
        "disposition": "Petition Denied",
        "votes_for": 4,
        "votes_against": 0,
        "opinion_text": """In all Cases affecting Ambassadors, other public Ministers and Consuls, and those in which a State shall be a Party, the supreme Court shall have original Jurisdiction. In all the other Cases before mentioned, the supreme Court shall have appellate Jurisdiction, both as to Law and Fact, with such Exceptions, and under such Regulations as the Congress shall make.

The authority, therefore, given to the Supreme Court by the act establishing the judicial courts of the United States to issue writs of mandamus to public officers appears not to be warranted by the Constitution.

It is emphatically the province and duty of the judicial department to say what the law is. Those who apply the rule to particular cases, must of necessity expound and interpret that rule. If two laws conflict with each other, the courts must decide on the operation of each.""",
        "source": "preserved"
    },
    {
        "case_name": "McCulloch v. Maryland",
        "citation": "17 U.S. 316",
        "term": "1819",
        "chief_justice": "John Marshall",
        "majority_author": "John Marshall",
        "word_count": 6800,
        "topic": "Federalism",
        "disposition": "Reversed",
        "votes_for": 7,
        "votes_against": 0,
        "opinion_text": """Let the end be legitimate, let it be within the scope of the Constitution, and all means which are appropriate, which are plainly adapted to that end, which are not prohibited, but consistent with the letter and spirit of the Constitution, are constitutional.

The power of creating a corporation is never used for its own sake, but for the purpose of effecting something else. No sufficient reason is perceived why it may not pass as incidental to those powers which are expressly given if it be a direct mode of executing them.""",
        "source": "preserved"
    },
    {
        "case_name": "Gibbons v. Ogden",
        "citation": "22 U.S. 1",
        "term": "1824",
        "chief_justice": "John Marshall",
        "majority_author": "John Marshall",
        "word_count": 5500,
        "topic": "Commerce Clause",
        "disposition": "Reversed",
        "votes_for": 6,
        "votes_against": 0,
        "opinion_text": """The power to regulate commerce extends to every species of commercial intercourse between the United States and foreign nations and among the several States. It does not stop at the external boundary of a State.

The power over commerce, including navigation, was vested entirely in Congress. The power of regulating commerce extends to the regulation of navigation.""",
        "source": "preserved"
    },
    {
        "case_name": "Dred Scott v. Sandford",
        "citation": "60 U.S. 393",
        "term": "1856",
        "chief_justice": "Roger B. Taney",
        "majority_author": "Roger B. Taney",
        "word_count": 12500,
        "topic": "Civil Rights",
        "disposition": "Dismissed",
        "votes_for": 7,
        "votes_against": 2,
        "opinion_text": """We think they are not, and that they are not included, and were not intended to be included, under the word 'citizens' in the Constitution, and can therefore claim none of the rights and privileges which that instrument provides for and secures to citizens of the United States. On the contrary, they were at that time considered as a subordinate and inferior class of beings who had been subjugated by the dominant race.

The right of property in a slave is distinctly and expressly affirmed in the Constitution. The Federal Government has no power to regulate slavery in the Territories.""",
        "source": "preserved"
    },
    {
        "case_name": "Plessy v. Ferguson",
        "citation": "163 U.S. 537",
        "term": "1896",
        "chief_justice": "Melville Fuller",
        "majority_author": "Henry Billings Brown",
        "word_count": 5800,
        "topic": "Civil Rights",
        "disposition": "Affirmed",
        "votes_for": 7,
        "votes_against": 1,
        "opinion_text": """The object of the [Fourteenth] Amendment was undoubtedly to enforce the absolute equality of the two races before the law, but in the nature of things it could not have been intended to abolish distinctions based upon color, or to enforce social, as distinguished from political, equality, or a commingling of the two races upon terms unsatisfactory to either.

We consider the underlying fallacy of the plaintiff's argument to consist in the assumption that the enforced separation of the two races stamps the colored race with a badge of inferiority. If this be so, it is not by reason of anything found in the act, but solely because the colored race chooses to put that construction upon it.""",
        "source": "preserved"
    },
    {
        "case_name": "Schenck v. United States",
        "citation": "249 U.S. 47",
        "term": "1918",
        "chief_justice": "Edward Douglass White",
        "majority_author": "Oliver Wendell Holmes Jr.",
        "word_count": 2100,
        "topic": "First Amendment",
        "disposition": "Affirmed",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": """The most stringent protection of free speech would not protect a man in falsely shouting fire in a theatre and causing a panic. It does not even protect a man from an injunction against uttering words that may have all the effect of force.

The question in every case is whether the words used are used in such circumstances and are of such a nature as to create a clear and present danger that they will bring about the substantive evils that Congress has a right to prevent. It is a question of proximity and degree.""",
        "source": "preserved"
    },
    {
        "case_name": "Gitlow v. New York",
        "citation": "268 U.S. 652",
        "term": "1923",
        "chief_justice": "William Howard Taft",
        "majority_author": "Edward Sanford",
        "word_count": 4500,
        "topic": "First Amendment",
        "disposition": "Affirmed",
        "votes_for": 7,
        "votes_against": 2,
        "opinion_text": """For present purposes we may and do assume that freedom of speech and of the press which are protected by the First Amendment from abridgment by Congress are among the fundamental personal rights and 'liberties' protected by the due process clause of the Fourteenth Amendment from impairment by the States.

The general principle of free speech, that it may be restrained when its expression is inimical to the public welfare, must be applied in consonance with the constitutional guarantee.""",
        "source": "preserved"
    },
    {
        "case_name": "Near v. Minnesota",
        "citation": "283 U.S. 697",
        "term": "1929",
        "chief_justice": "Charles Evans Hughes",
        "majority_author": "Charles Evans Hughes",
        "word_count": 6200,
        "topic": "First Amendment",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The liberty of the press has been especially cherished in English-speaking countries because it has been recognized as the best safeguard of all other liberties. The fact that the liberty of the press may be abused by miscreant purveyors of scandal does not make any the less necessary the immunity of the press from previous restraint.

The chief purpose of the guaranty is to prevent previous restraints upon publication. The fact that the liberty of the press may be abused by miscreant purveyors of scandal does not make any the less necessary the immunity of the press from previous restraint.""",
        "source": "preserved"
    },
    {
        "case_name": "West Coast Hotel v. Parrish",
        "citation": "300 U.S. 379",
        "term": "1936",
        "chief_justice": "Charles Evans Hughes",
        "majority_author": "Charles Evans Hughes",
        "word_count": 3500,
        "topic": "Economic Regulation",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The Constitution does not speak of freedom of contract. It speaks of liberty and prohibits the deprivation of liberty without due process of law. In prohibiting that deprivation the Constitution does not recognize an absolute and uncontrollable liberty.

The legislature has the right to consider that the community may be adversely affected by the existence of persons who are too poor to maintain decent standards of health and morals. The community is not bound to provide what is in effect a subsidy for unconscionable employers.""",
        "source": "preserved"
    },
    {
        "case_name": "United States v. Darby",
        "citation": "312 U.S. 100",
        "term": "1940",
        "chief_justice": "Charles Evans Hughes",
        "majority_author": "Harlan Fiske Stone",
        "word_count": 4800,
        "topic": "Commerce Clause",
        "disposition": "Affirmed",
        "votes_for": 8,
        "votes_against": 0,
        "opinion_text": """The power of Congress over interstate commerce is not confined to the regulation of commerce among the states. It extends to those activities intrastate which so affect interstate commerce or the exercise of the power of Congress over it as to make regulation of them appropriate means to the attainment of a legitimate end.

The power of Congress over interstate commerce extends to those activities intrastate which so affect interstate commerce or the exercise of the power of Congress over it as to make regulation of them appropriate means to the attainment of a legitimate end.""",
        "source": "preserved"
    },
    {
        "case_name": "Wickard v. Filburn",
        "citation": "317 U.S. 111",
        "term": "1942",
        "chief_justice": "Harlan Fiske Stone",
        "majority_author": "Robert H. Jackson",
        "word_count": 5100,
        "topic": "Commerce Clause",
        "disposition": "Reversed",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": """Even if appellee's activity be local and though it may not be regarded as commerce, it may still, whatever its nature, be reached by Congress if it exerts a substantial economic effect on interstate commerce.

Whether the subject of the regulation in question was 'production', 'consumption', or 'marketing' is, therefore, not material for purposes of deciding the question of federal power before us. That an activity is of local character may help in answering whether it exerts a substantial effect on interstate commerce.""",
        "source": "preserved"
    },
    {
        "case_name": "Korematsu v. United States",
        "citation": "323 U.S. 214",
        "term": "1944",
        "chief_justice": "Harlan Fiske Stone",
        "majority_author": "Hugo Black",
        "word_count": 3900,
        "topic": "Civil Rights",
        "disposition": "Affirmed",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": """It should be noted, to begin with, that all legal restrictions which curtail the civil rights of a single racial group are immediately suspect. That is not to say that all such restrictions are unconstitutional. It is to say that courts must subject them to the most rigid scrutiny.

Pressing public necessity may sometimes justify the existence of such restrictions; racial antagonism never can.

Compulsory exclusion, large or small, is not justified.""",
        "source": "preserved"
    },
    {
        "case_name": "Brown v. Board of Education",
        "citation": "347 U.S. 483",
        "term": "1953",
        "chief_justice": "Earl Warren",
        "majority_author": "Earl Warren",
        "word_count": 2800,
        "topic": "Civil Rights",
        "disposition": "Reversed",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": """We conclude that in the field of public education the doctrine of 'separate but equal' has no place. Separate educational facilities are inherently unequal.

We conclude that in the field of public education the doctrine of 'separate but equal' has no place. Separate educational facilities are inherently unequal. Therefore, we hold that the plaintiffs and others similarly situated for whom the actions have been brought are, by reason of the segregation complained of, deprived of the equal protection of the laws guaranteed by the Fourteenth Amendment.

In these days, it is doubtful that any child may reasonably be expected to succeed in life if he is denied the opportunity of an education. Such an opportunity, where the state has undertaken to provide it, is a right which must be made available to all on equal terms.""",
        "source": "preserved"
    },
    {
        "case_name": "Mapp v. Ohio",
        "citation": "367 U.S. 643",
        "term": "1960",
        "chief_justice": "Earl Warren",
        "majority_author": "Tom C. Clark",
        "word_count": 4700,
        "topic": "Fourth Amendment",
        "disposition": "Reversed",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": """We hold that all evidence obtained by searches and seizures in violation of the Constitution is, by that same authority, inadmissible in a state court.

The ignoble shortcut to conviction left open to the State tends to destroy the entire system of constitutional restraints on which the liberties of the people rest. Having once recognized that the right to privacy embodied in the Fourth Amendment is enforceable against the States, and that the right to be secure against rude invasions of privacy by state officers is, therefore, constitutional in origin, we can no longer permit that right to remain an empty promise.""",
        "source": "preserved"
    },
    {
        "case_name": "Engel v. Vitale",
        "citation": "370 U.S. 421",
        "term": "1962",
        "chief_justice": "Earl Warren",
        "majority_author": "Hugo Black",
        "word_count": 3200,
        "topic": "First Amendment",
        "disposition": "Reversed",
        "votes_for": 6,
        "votes_against": 1,
        "opinion_text": """It is no part of the business of government to compose official prayers for any group of the American people to recite as a part of a religious program carried on by government.

The Establishment Clause does not depend upon any showing that compelled religious exercise or the involuntary exertion of religious influence is the necessary result of the challenged governmental action. The very purpose of the Establishment Clause is to prevent the government from taking sides in religious matters.""",
        "source": "preserved"
    },
    {
        "case_name": "Gideon v. Wainwright",
        "citation": "372 U.S. 335",
        "term": "1962",
        "chief_justice": "Earl Warren",
        "majority_author": "Hugo Black",
        "word_count": 2600,
        "topic": "Sixth Amendment",
        "disposition": "Reversed",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": """The right of one charged with crime to counsel may not be deemed fundamental and essential to fair trials in some countries, but it is in ours.

Reason and reflection require us to recognize that in our adversary system of criminal justice, any person haled into court, who is too poor to hire a lawyer, cannot be assured a fair trial unless counsel is provided for him. This seems to us to be an obvious truth.""",
        "source": "preserved"
    },
    {
        "case_name": "New York Times v. Sullivan",
        "citation": "376 U.S. 254",
        "term": "1963",
        "chief_justice": "Earl Warren",
        "majority_author": "William J. Brennan Jr.",
        "word_count": 8900,
        "topic": "First Amendment",
        "disposition": "Reversed",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": """The constitutional guarantees require, we think, a federal rule that prohibits a public official from recovering damages for a defamatory falsehood relating to his official conduct unless he proves that the statement was made with 'actual malice'—that is, with knowledge that it was false or with reckless disregard of whether it was false or not.

Thus we consider this case against the background of a profound national commitment to the principle that debate on public issues should be uninhibited, robust, and wide-open, and that it may well include vehement, caustic, and sometimes unpleasantly sharp attacks on government and public officials.""",
        "source": "preserved"
    },
    {
        "case_name": "Miranda v. Arizona",
        "citation": "384 U.S. 436",
        "term": "1965",
        "chief_justice": "Earl Warren",
        "majority_author": "Earl Warren",
        "word_count": 11200,
        "topic": "Fifth Amendment",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The person in custody must, prior to interrogation, be clearly informed that he has the right to remain silent, and that anything he says will be used against him in court; he must be clearly informed that he has the right to consult with a lawyer and to have the lawyer with him during interrogation, and that, if he is indigent, a lawyer will be appointed to represent him.

We hold that when an individual is taken into custody or otherwise deprived of his freedom by the authorities in any significant way and is subjected to questioning, the privilege against self-incrimination is jeopardized. Procedural safeguards must be employed to protect the privilege, and unless other fully effective means are adopted to notify the person of his right of silence and to assure that the exercise of the right will be scrupulously honored, the following measures are required.""",
        "source": "preserved"
    },
    {
        "case_name": "Brandenburg v. Ohio",
        "citation": "395 U.S. 444",
        "term": "1968",
        "chief_justice": "Earl Warren",
        "majority_author": "William J. Brennan Jr.",
        "word_count": 1800,
        "topic": "First Amendment",
        "disposition": "Reversed",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": """The constitutional guarantees of free speech and free press do not permit a State to forbid or proscribe advocacy of the use of force or of law violation except where such advocacy is directed to inciting or producing imminent lawless action and is likely to incite or produce such action.

A statute which fails to draw this distinction impermissibly intrudes upon the freedoms guaranteed by the First and Fourteenth Amendments. It sweeps within its condemnation speech which our Constitution has immunized from governmental control.""",
        "source": "preserved"
    },
    {
        "case_name": "Roe v. Wade",
        "citation": "410 U.S. 113",
        "term": "1971",
        "chief_justice": "Warren E. Burger",
        "majority_author": "Harry Blackmun",
        "word_count": 14500,
        "topic": "Abortion",
        "disposition": "Reversed",
        "votes_for": 7,
        "votes_against": 2,
        "opinion_text": """We forthwith acknowledge our awareness of the sensitive and emotional nature of the abortion controversy, of the vigorous opposing views, even among physicians, and of the deep and seemingly absolute convictions that the subject inspires. One's philosophy, one's experiences, one's exposure to the raw edges of human existence, one's religious training, one's attitudes toward life and family and their values, and the moral standards one establishes and seeks to observe, are all likely to influence and to color one's thinking and conclusions about abortion.

The Constitution does not explicitly mention any right of privacy. In a line of decisions, however, going back perhaps as far as Union Pacific R. Co. v. Botsford, the Court has recognized that a right of personal privacy, or a guarantee of certain areas or zones of privacy, does exist under the Constitution.

This right of privacy, whether it be founded in the Fourteenth Amendment's concept of personal liberty and restrictions upon state action, as we feel it is, or, as the District Court determined, in the Ninth Amendment's reservation of rights to the people, is broad enough to encompass a woman's decision whether or not to terminate her pregnancy.""",
        "source": "preserved"
    },
    {
        "case_name": "United States v. Nixon",
        "citation": "418 U.S. 683",
        "term": "1973",
        "chief_justice": "Warren E. Burger",
        "majority_author": "Warren E. Burger",
        "word_count": 8200,
        "topic": "Executive Power",
        "disposition": "Affirmed",
        "votes_for": 8,
        "votes_against": 0,
        "opinion_text": """Neither the doctrine of separation of powers, nor the need for confidentiality of high-level communications, without more, can sustain an absolute, unqualified Presidential privilege of immunity from judicial process under all circumstances.

The impediment that an absolute, unqualified privilege would place in the way of the primary constitutional duty of the Judicial Branch to do justice in criminal prosecutions would plainly conflict with the function of the courts under Art. III.""",
        "source": "preserved"
    },
    {
        "case_name": "Regents v. Bakke",
        "citation": "438 U.S. 265",
        "term": "1977",
        "chief_justice": "Warren E. Burger",
        "majority_author": "Lewis Powell",
        "word_count": 9500,
        "topic": "Civil Rights",
        "disposition": "Reversed in part",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The guarantee of equal protection cannot mean one thing when applied to one individual and something else when applied to another. If both are not accorded the same protection, then it is not equal.

Preferring members of any one group for no reason other than race or ethnic origin is discrimination for its own sake. This the Constitution forbids.""",
        "source": "preserved"
    },
    {
        "case_name": "Texas v. Johnson",
        "citation": "491 U.S. 397",
        "term": "1988",
        "chief_justice": "William Rehnquist",
        "majority_author": "William J. Brennan Jr.",
        "word_count": 5400,
        "topic": "First Amendment",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """If there is a bedrock principle underlying the First Amendment, it is that the government may not prohibit the expression of an idea simply because society finds the idea itself offensive or disagreeable.

We do not consecrate the flag by punishing its desecration, for in doing so we dilute the freedom that this cherished emblem represents.""",
        "source": "preserved"
    },
    {
        "case_name": "Planned Parenthood v. Casey",
        "citation": "505 U.S. 833",
        "term": "1991",
        "chief_justice": "William Rehnquist",
        "majority_author": "Sandra Day O'Connor",
        "word_count": 12800,
        "topic": "Abortion",
        "disposition": "Affirmed in part, Reversed in part",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """Liberty finds no refuge in a jurisprudence of doubt. Yet 19 years after our holding that the Constitution protects a woman's right to terminate her pregnancy in its early stages, Roe v. Wade, that definition of liberty is still questioned.

The mother's liberty is not so broad as to make it impossible for the State to protect the life of the unborn. The State has legitimate interests from the outset of the pregnancy in protecting the health of the woman and the life of the fetus that may become a child.""",
        "source": "preserved"
    },
    {
        "case_name": "United States v. Lopez",
        "citation": "514 U.S. 549",
        "term": "1994",
        "chief_justice": "William Rehnquist",
        "majority_author": "William Rehnquist",
        "word_count": 6100,
        "topic": "Federalism",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The Constitution creates a Federal Government of enumerated powers. The Constitution delegates to Congress the power '[t]o regulate Commerce with foreign Nations, and among the several States, and with the Indian Tribes.'

To uphold the Government's contention that the Commerce Clause gives Congress the power to regulate activity that is not itself interstate commerce merely because it affects interstate commerce would require us to pile inference upon inference in a manner that would bid fair to convert congressional authority under the Commerce Clause to a general police power of the sort retained by the States.""",
        "source": "preserved"
    },
    {
        "case_name": "Bush v. Gore",
        "citation": "531 U.S. 98",
        "term": "2000",
        "chief_justice": "William Rehnquist",
        "majority_author": "William Rehnquist",
        "word_count": 4200,
        "topic": "Election Law",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The individual citizen has no federal constitutional right to vote for electors for the President of the United States unless and until the state legislature chooses a statewide election as the means to implement its power to appoint members of the Electoral College.

When the state legislature vests the right to vote for President in its people, the right to vote as the legislature has prescribed is fundamental; and one source of its fundamental nature lies in the equal weight accorded to each vote and the equal dignity owed to each voter.""",
        "source": "preserved"
    },
    {
        "case_name": "Lawrence v. Texas",
        "citation": "539 U.S. 558",
        "term": "2002",
        "chief_justice": "William Rehnquist",
        "majority_author": "Anthony Kennedy",
        "word_count": 6800,
        "topic": "Marriage Equality",
        "disposition": "Reversed",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": """Liberty presumes an autonomy of self that includes freedom of thought, belief, expression, and certain intimate conduct. The instant case involves liberty of the person both in its spatial and more transcendent dimensions.

The State cannot demean their existence or control their destiny by making their private sexual conduct a crime. Their right to liberty under the Due Process Clause gives them the full right to engage in their conduct without intervention of the government.""",
        "source": "preserved"
    },
    {
        "case_name": "District of Columbia v. Heller",
        "citation": "554 U.S. 570",
        "term": "2007",
        "chief_justice": "John Roberts",
        "majority_author": "Antonin Scalia",
        "word_count": 11800,
        "topic": "Gun Rights",
        "disposition": "Affirmed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The Second Amendment protects an individual right to possess a firearm unconnected with service in a militia, and to use that arm for traditionally lawful purposes, such as self-defense within the home.

The prefatory clause announces a purpose, but does not limit or expand the scope of the second part, the operative clause. The operative clause's text and history demonstrate that it connotes an individual right to keep and bear arms.""",
        "source": "preserved"
    },
    {
        "case_name": "Citizens United v. FEC",
        "citation": "558 U.S. 310",
        "term": "2008",
        "chief_justice": "John Roberts",
        "majority_author": "Anthony Kennedy",
        "word_count": 10200,
        "topic": "Election Law",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """If the First Amendment has any force, it prohibits Congress from fining or jailing citizens, or associations of citizens, for simply engaging in political speech.

The government may regulate corporate political speech through disclaimer and disclosure requirements, but it may not suppress that speech altogether.""",
        "source": "preserved"
    },
    {
        "case_name": "National Federation of Independent Business v. Sebelius",
        "citation": "567 U.S. 519",
        "term": "2011",
        "chief_justice": "John Roberts",
        "majority_author": "John Roberts",
        "word_count": 15600,
        "topic": "Health Care",
        "disposition": "Affirmed in part, Reversed in part",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The Federal Government does not have the power to order people to buy health insurance. Section 5000A would therefore be unconstitutional if read as a command.

The Federal Government does have the power to impose a tax on those without health insurance. Section 5000A is therefore constitutional because it can reasonably be read as a tax.""",
        "source": "preserved"
    },
    {
        "case_name": "United States v. Windsor",
        "citation": "570 U.S. 744",
        "term": "2012",
        "chief_justice": "John Roberts",
        "majority_author": "Anthony Kennedy",
        "word_count": 7400,
        "topic": "Marriage Equality",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """DOMA seeks to injure the very class New York seeks to protect. By doing so it violates basic due process and equal protection principles applicable to the Federal Government.

The federal statute is invalid, for no legitimate purpose overcomes the purpose and effect to disparage and to injure those whom the State, by its marriage laws, sought to protect in personhood and dignity.""",
        "source": "preserved"
    },
    {
        "case_name": "Obergefell v. Hodges",
        "citation": "576 U.S. 644",
        "term": "2014",
        "chief_justice": "John Roberts",
        "majority_author": "Anthony Kennedy",
        "word_count": 9200,
        "topic": "Marriage Equality",
        "disposition": "Reversed",
        "votes_for": 5,
        "votes_against": 4,
        "opinion_text": """The right to marry is fundamental as a matter of history and tradition, but rights come not from ancient sources alone. They rise, too, from a better informed understanding of how constitutional imperatives define a liberty that remains urgent in our own era.

No union is more profound than marriage, for it embodies the highest ideals of love, fidelity, devotion, sacrifice, and family. In forming a marital union, two people become something greater than once they were. As some of the petitioners in these cases demonstrate, marriage embodies a love that may endure even past death. It would misunderstand these men and women to say they disrespect the idea of marriage. Their plea is that they do respect it, respect it so deeply that they seek to find its fulfillment for themselves.""",
        "source": "preserved"
    },
    {
        "case_name": "Youngstown Sheet & Tube Co. v. Sawyer",
        "citation": "343 U.S. 579",
        "term": "1952",
        "chief_justice": "Fred M. Vinson",
        "majority_author": "Hugo Black",
        "word_count": 4600,
        "topic": "Executive Power",
        "disposition": "Reversed",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": """The President's power, if any, to issue the order must stem either from an act of Congress or from the Constitution itself. There is no statute that expressly authorizes the President to take possession of property as a result of a labor dispute.

In the framework of our Constitution, the President's power to see that the laws are faithfully executed refutes the idea that he is to be a lawmaker. The Constitution limits his functions in the lawmaking process to the recommending of laws he thinks wise and the vetoing of laws he thinks bad.""",
        "source": "preserved"
    },
    {
        "case_name": "Griswold v. Connecticut",
        "citation": "381 U.S. 479",
        "term": "1965",
        "chief_justice": "Earl Warren",
        "majority_author": "William O. Douglas",
        "word_count": 3200,
        "topic": "Fourteenth Amendment",
        "disposition": "Reversed",
        "votes_for": 7,
        "votes_against": 2,
        "opinion_text": """Would we allow the police to search the sacred precincts of marital bedrooms for telltale signs of the use of contraceptives? The very idea is repulsive to the notions of privacy surrounding the marriage relationship.

We deal with a right of privacy older than the Bill of Rights—older than our political parties, older than our school system. Marriage is a coming together for better or for worse, hopefully enduring, and intimate to the degree of being sacred.""",
        "source": "preserved"
    },
    {
        "case_name": "Tinker v. Des Moines",
        "citation": "393 U.S. 503",
        "term": "1968",
        "chief_justice": "Earl Warren",
        "majority_author": "Abe Fortas",
        "word_count": 3600,
        "topic": "First Amendment",
        "disposition": "Affirmed",
        "votes_for": 7,
        "votes_against": 2,
        "opinion_text": """It can hardly be argued that either students or teachers shed their constitutional rights to freedom of speech or expression at the schoolhouse gate.

In the absence of a specific showing of constitutionally valid reasons to regulate their speech, students are entitled to freedom of expression of their views.""",
        "source": "preserved"
    },
    {
        "case_name": "Heart of Atlanta Motel v. United States",
        "citation": "379 U.S. 241",
        "term": "1964",
        "chief_justice": "Earl Warren",
        "majority_author": "Tom C. Clark",
        "word_count": 5400,
        "topic": "Commerce Clause",
        "disposition": "Affirmed",
        "votes_for": 9,
        "votes_against": 0,
        "opinion_text": """The power of Congress to keep interstate commerce free from immoral and injurious intrusions has been firmly established. The Civil Rights Act of 1964 is a valid exercise of that power.

The motel's operations affect interstate commerce and therefore Congress has the power to regulate them under the Commerce Clause.""",
        "source": "preserved"
    },
    {
        "case_name": "New York Times Co. v. United States",
        "citation": "403 U.S. 713",
        "term": "1970",
        "chief_justice": "Warren E. Burger",
        "majority_author": "Per Curiam",
        "word_count": 2100,
        "topic": "First Amendment",
        "disposition": "Affirmed",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": """The Government carries a heavy burden of showing justification for the enforcement of such a restraint. That burden has not been met. Any system of prior restraints of expression comes to this Court bearing a heavy presumption against its constitutional validity.

The word 'security' is a broad, vague generality whose contours should not be invoked to abrogate the fundamental law embodied in the First Amendment.""",
        "source": "preserved"
    },
    {
        "case_name": "Buckley v. Valeo",
        "citation": "424 U.S. 1",
        "term": "1975",
        "chief_justice": "Warren E. Burger",
        "majority_author": "Per Curiam",
        "word_count": 16800,
        "topic": "Election Law",
        "disposition": "Affirmed in part, Reversed in part",
        "votes_for": 8,
        "votes_against": 1,
        "opinion_text": """The Act's contribution and expenditure limitations operate in an area of the most fundamental First Amendment activities. Discussion of public issues and debate on the qualifications of candidates are integral to the operation of the system of government established by our Constitution.

A restriction on the amount of money a person or group can spend on political communication during a campaign necessarily reduces the quantity of expression by restricting the number of issues discussed, the depth of their exploration, and the size of the audience reached.""",
        "source": "preserved"
    },
    {
        "case_name": "Roberts v. United States Jaycees",
        "citation": "468 U.S. 609",
        "term": "1983",
        "chief_justice": "Warren E. Burger",
        "majority_author": "William J. Brennan Jr.",
        "word_count": 6100,
        "topic": "Civil Rights",
        "disposition": "Reversed",
        "votes_for": 7,
        "votes_against": 0,
        "opinion_text": """An individual's freedom to speak, to worship, and to petition the government for the redress of grievances could not be vigorously protected from interference by the State unless a correlative freedom to engage in group efforts toward those ends were also safeguarded.

The right to associate for expressive purposes is not, however, absolute. Infringements on that right may be justified by regulations adopted to serve compelling state interests, unrelated to the suppression of ideas, that cannot be achieved through means significantly less restrictive of associational freedoms.""",
        "source": "preserved"
    },
    {
        "case_name": "City of Boerne v. Flores",
        "citation": "521 U.S. 507",
        "term": "1996",
        "chief_justice": "William Rehnquist",
        "majority_author": "Anthony Kennedy",
        "word_count": 6800,
        "topic": "Fourteenth Amendment",
        "disposition": "Reversed",
        "votes_for": 6,
        "votes_against": 3,
        "opinion_text": """Legislation which alters the meaning of the Free Exercise Clause cannot be said to be enforcing the Clause. Congress does not enforce a constitutional right by changing what the right is.

While preventive rules are sometimes appropriate remedial measures, there must be a congruence between the means used and the ends to be achieved. The appropriateness of remedial measures must be considered in light of the evil presented.""",
        "source": "preserved"
    },
]


def get_preserved_cases() -> List[Dict]:
    """Return preserved landmark cases as ultimate fallback."""
    return [dict(c) for c in PRESERVED_LANDMARK_CASES]


# ── Main Orchestrator ─────────────────────────────────────────────────────

def fetch_all_scotus_data(target_count: int = 30) -> List[Dict]:
    """
    Fetch SCOTUS case data using live APIs with fallback to preserved data.

    Strategy:
      1. Try CourtListener (if token available)
      2. Try Oyez for landmark cases
      3. Fill remaining slots with preserved landmark data
    """
    cases = []
    sources = {'courtlistener': 0, 'oyez': 0, 'preserved': 0}

    # 1. Try CourtListener
    cl_cases = fetch_courtlistener_opinions(max_cases=target_count)
    for c in cl_cases:
        cases.append(c)
        sources['courtlistener'] += 1
    logger.info("CourtListener: fetched %d cases", len(cl_cases))

    # 2. Try Oyez
    if len(cases) < target_count:
        oyez_needed = target_count - len(cases)
        oyez_cases = fetch_oyez_landmark_cases(max_cases=oyez_needed + 5)
        for c in oyez_cases:
            # Avoid duplicates by case name
            if not any(existing['case_name'] == c['case_name'] for existing in cases):
                cases.append(c)
                sources['oyez'] += 1
                if len(cases) >= target_count:
                    break
        logger.info("Oyez: fetched %d new cases", len(oyez_cases))

    # 3. Fill with preserved data
    if len(cases) < target_count:
        preserved = get_preserved_cases()
        for c in preserved:
            if not any(existing['case_name'] == c['case_name'] for existing in cases):
                cases.append(c)
                sources['preserved'] += 1
                if len(cases) >= target_count:
                    break
        logger.info("Preserved: added %d cases", len(cases) - sources['courtlistener'] - sources['oyez'])

    logger.info("Total cases: %d (CourtListener: %d, Oyez: %d, Preserved: %d)",
                len(cases), sources['courtlistener'], sources['oyez'], sources['preserved'])
    return cases


# ── Visualization (same as original) ────────────────────────────────────────

def create_figures(cases: List[Dict], output_dir: Path) -> None:
    """Create visualization figures from case data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Word count distribution
    word_counts = [c['word_count'] for c in cases if c.get('word_count')]
    if word_counts:
        plt.figure(figsize=(10, 6))
        plt.hist(word_counts, bins=15, edgecolor='black')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of SCOTUS Opinion Word Counts')
        plt.tight_layout()
        plt.savefig(output_dir / 'word_count_distribution.png')
        plt.close()

    # Figure 2: Cases by topic
    topics = {}
    for c in cases:
        t = c.get('topic', 'Unknown')
        topics[t] = topics.get(t, 0) + 1

    if topics:
        plt.figure(figsize=(12, 6))
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        names, counts = zip(*sorted_topics)
        plt.barh(range(len(names)), counts)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Number of Cases')
        plt.title('SCOTUS Cases by Topic')
        plt.tight_layout()
        plt.savefig(output_dir / 'cases_by_topic.png')
        plt.close()

    # Figure 3: Votes distribution
    vote_diffs = []
    for c in cases:
        vf = c.get('votes_for', 0)
        va = c.get('votes_against', 0)
        if vf or va:
            vote_diffs.append(vf - va)

    if vote_diffs:
        plt.figure(figsize=(10, 6))
        plt.hist(vote_diffs, bins=range(-10, 11), edgecolor='black')
        plt.xlabel('Vote Margin (For - Against)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Vote Margins')
        plt.tight_layout()
        plt.savefig(output_dir / 'vote_margins.png')
        plt.close()

    # Figure 4: Cases over time
    terms = {}
    for c in cases:
        t = c.get('term', '')
        if t:
            try:
                year = int(t)
                decade = (year // 10) * 10
                terms[decade] = terms.get(decade, 0) + 1
            except ValueError:
                pass

    if terms:
        plt.figure(figsize=(12, 6))
        sorted_terms = sorted(terms.items())
        decades, counts = zip(*sorted_terms)
        plt.bar([str(d) + 's' for d in decades], counts)
        plt.xlabel('Decade')
        plt.ylabel('Number of Cases')
        plt.title('SCOTUS Cases by Decade')
        plt.tight_layout()
        plt.savefig(output_dir / 'cases_by_decade.png')
        plt.close()

    logger.info("Figures saved to %s", output_dir)


# ── Entry Point ───────────────────────────────────────────────────────────

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Fetch SCOTUS opinion data')
    parser.add_argument('--count', type=int, default=30, help='Number of cases to fetch')
    parser.add_argument('--output', type=str, default='data/scotus_cases.json',
                        help='Output JSON file path')
    parser.add_argument('--figures', type=str, default='data/figures',
                        help='Output directory for figures')
    parser.add_argument('--courtlistener-token', type=str, default='',
                        help='CourtListener API token (or set COURTLISTENER_API_TOKEN env var)')
    args = parser.parse_args()

    # Set token from CLI if provided
    global COURTLISTENER_TOKEN
    if args.courtlistener_token:
        COURTLISTENER_TOKEN = args.courtlistener_token

    logger.info("Starting SCOTUS data fetch (target: %d cases)", args.count)
    logger.info("CourtListener token: %s", "configured" if COURTLISTENER_TOKEN else "not configured")

    cases = fetch_all_scotus_data(target_count=args.count)

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d cases to %s", len(cases), output_path)

    # Create figures
    create_figures(cases, args.figures)

    # Print summary
    sources = {}
    for c in cases:
        s = c.get('source', 'unknown')
        sources[s] = sources.get(s, 0) + 1

    print("\n" + "="*60)
    print("SCOTUS DATA FETCH COMPLETE")
    print("="*60)
    print(f"Total cases: {len(cases)}")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count}")
    print(f"Output: {args.output}")
    print(f"Figures: {args.figures}/")
    print("="*60)

    return cases


if __name__ == '__main__':
    main()
