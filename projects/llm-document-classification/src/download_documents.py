"""
Live API Document Downloader for Multi-Domain Text Classification
=================================================================
Fetches real documents from public APIs across 4 domains:
  - ArXiv (scientific/technical + financial abstracts) — export.arxiv.org/api/query
  - PubMed (medical/health abstracts) — eutils.ncbi.nlm.nih.gov
  - Wikipedia (legal/financial/general encyclopedia articles) — en.wikipedia.org/api/rest_v1/

Target: ≥1,000 documents across ≥3 categories.
"""

import os
import time
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; mailto:research@example.edu)"
}


# ---------------------------------------------------------------------------
# ArXiv fetcher
# ---------------------------------------------------------------------------

def fetch_arxiv(max_results: int = 600) -> List[Dict]:
    """Fetch ArXiv abstracts via the public Atom API.
    Source: https://arxiv.org/help/api/index
    Categories: cs.*, physics.*, math.*, q-bio.*, stat.*, econ.*, q-fin.*
    """
    records = []
    base_url = "http://export.arxiv.org/api/query"

    queries = [
        ("cat:cs.*", "scientific"),
        ("cat:physics.*", "scientific"),
        ("cat:math.*", "scientific"),
        ("cat:q-bio.*", "scientific"),
        ("cat:stat.*", "scientific"),
        ("cat:econ.*", "financial"),
        ("cat:q-fin.*", "financial"),
        ("cat:astro-ph.*", "scientific"),
        ("cat:cond-mat.*", "scientific"),
    ]

    per_query = max(20, max_results // len(queries))

    for query, label in queries:
        if len(records) >= max_results:
            break
        fetch_n = min(per_query, max_results - len(records))
        url = (
            f"{base_url}?search_query={query}"
            f"&start=0&max_results={fetch_n}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[ArXiv] Error fetching {query}: {exc}")
            continue

        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            id_url = entry.find("atom:id", ns)
            published = entry.find("atom:published", ns)

            text_parts = []
            if title is not None and title.text:
                text_parts.append(title.text.strip())
            if summary is not None and summary.text:
                text_parts.append(summary.text.strip())

            if not text_parts:
                continue

            records.append({
                "doc_id": id_url.text.strip().split("/")[-1] if id_url is not None else f"arxiv_{len(records)}",
                "source": "arxiv",
                "category": label,
                "subcategory": query,
                "title": title.text.strip() if title is not None else "",
                "text": "\n\n".join(text_parts),
                "url": id_url.text.strip() if id_url is not None else "",
                "date": published.text.strip() if published is not None else "",
            })

        time.sleep(3)

    print(f"[ArXiv] Fetched {len(records)} abstracts")
    return records


# ---------------------------------------------------------------------------
# PubMed fetcher
# ---------------------------------------------------------------------------

def fetch_pubmed(max_results: int = 450) -> List[Dict]:
    """Fetch PubMed abstracts via NCBI E-utilities.
    Source: https://www.ncbi.nlm.nih.gov/books/NBK25500/
    """
    records = []
    esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    terms = [
        ("machine+learning+diagnosis", "medical"),
        ("public+health+intervention", "medical"),
        ("clinical+trial+outcome", "medical"),
        ("epidemiology+infectious+disease", "medical"),
        ("health+policy+evaluation", "medical"),
        ("cancer+treatment+protocol", "medical"),
        ("cardiovascular+risk+assessment", "medical"),
        ("mental+health+therapy", "medical"),
        ("diabetes+management", "medical"),
        ("neurodegenerative+disease", "medical"),
    ]

    per_query = max(20, max_results // len(terms))

    for term, label in terms:
        if len(records) >= max_results:
            break
        fetch_n = min(per_query, max_results - len(records))

        search_params = {
            "db": "pubmed",
            "term": term,
            "retmax": fetch_n,
            "retmode": "json",
            "sort": "pub+date",
        }
        try:
            r = requests.get(esearch, params=search_params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            pmids = r.json().get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            print(f"[PubMed] Search error for '{term}': {exc}")
            continue

        if not pmids:
            continue

        batch_size = 20
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }
            try:
                r2 = requests.get(efetch, params=fetch_params, headers=HEADERS, timeout=30)
                r2.raise_for_status()
            except Exception as exc:
                print(f"[PubMed] Fetch error: {exc}")
                continue

            root = ET.fromstring(r2.content)

            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                title_el = article.find(".//ArticleTitle")
                abstract_el = article.find(".//Abstract/AbstractText")
                date_el = article.find(".//PubDate/Year")

                pmid = pmid_el.text if pmid_el is not None else ""
                title = title_el.text if title_el is not None else ""
                abstract = abstract_el.text if abstract_el is not None else ""
                year = date_el.text if date_el is not None else ""

                text_parts = [p for p in [title, abstract] if p]
                if not text_parts:
                    continue

                records.append({
                    "doc_id": f"pubmed_{pmid}",
                    "source": "pubmed",
                    "category": label,
                    "subcategory": term,
                    "title": title,
                    "text": "\n\n".join(text_parts),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    "date": year,
                })

            time.sleep(1)

        time.sleep(1)

    print(f"[PubMed] Fetched {len(records)} abstracts")
    return records


# ---------------------------------------------------------------------------
# Wikipedia fetcher (legal / financial / general articles)
# ---------------------------------------------------------------------------

def fetch_wikipedia(max_results: int = 400) -> List[Dict]:
    """Fetch Wikipedia articles via the MediaWiki API and REST summary endpoint.
    Source: https://www.mediawiki.org/wiki/API:Main_page
    Uses page summaries from the REST API for clean text extraction.
    Also fetches related pages via the 'links' API to expand the corpus.
    """
    records = []
    seen_titles = set()

    titles_legal = [
        "Constitutional law", "Contract law", "Tort law", "Criminal law",
        "Securities regulation in the United States", "Antitrust",
        "Environmental law", "Civil procedure", "Intellectual property",
        "Corporate law", "Bankruptcy", "Immigration law",
        "Employment law", "Consumer protection", "Data protection",
        "International law", "Human rights", "Tax law",
        "Evidence law", "Family law", "Property law",
        "Administrative law", "Merger regulation", "Trade regulation",
        "Labor law", "Medical malpractice", "Product liability",
    ]
    titles_financial = [
        "Financial statement", "Balance sheet", "Income statement",
        "Cash flow", "Stock market", "Bond market", "Derivatives market",
        "Monetary policy", "Fiscal policy", "Investment banking",
        "Venture capital", "Private equity", "Hedge fund",
        "Exchange rate", "Interest rate", "Inflation",
        "Federal Reserve", "Central bank", "Commercial bank",
        "Accounting standard", "Audit", "Risk management",
        "Portfolio management", "Asset pricing", "Corporate finance",
        "Capital market", "Mergers and acquisitions", "Initial public offering",
        "Financial regulation", "Credit risk", "Market liquidity",
    ]

    all_titles = [(t, "legal") for t in titles_legal] + [(t, "financial") for t in titles_financial]
    random.shuffle(all_titles)

    def fetch_one(title: str, label: str) -> Dict:
        """Fetch a single Wikipedia article."""
        rest_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        try:
            resp = requests.get(rest_url, headers=HEADERS, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except Exception:
            return None

        data = resp.json()
        extract_text = data.get("extract", "")
        if len(extract_text) < 100:
            return None

        # Also try to fetch fuller text via extracts API
        raw_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&titles={title.replace(' ', '%20')}"
            "&explaintext=true&exlimit=1&format=json"
        )
        try:
            raw_resp = requests.get(raw_url, headers=HEADERS, timeout=15)
            raw_resp.raise_for_status()
            raw_data = raw_resp.json()
            pages = raw_data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                full_extract = page.get("extract", "")
                if full_extract and len(full_extract) > len(extract_text):
                    extract_text = full_extract
        except Exception:
            pass

        extract_text = extract_text[:2500]

        return {
            "doc_id": f"wiki_{title.replace(' ', '_').lower()}",
            "source": "wikipedia",
            "category": label,
            "subcategory": title,
            "title": data.get("title", title),
            "text": extract_text,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
            "date": "",
        }

    # First pass: fetch all primary titles
    for title, label in all_titles:
        if len(records) >= max_results:
            break
        if title in seen_titles:
            continue
        rec = fetch_one(title, label)
        if rec:
            records.append(rec)
            seen_titles.add(title)
        time.sleep(0.3)

    # Second pass: fetch related pages via links API to expand the corpus
    if len(records) < max_results:
        link_api = "https://en.wikipedia.org/w/api.php"
        for title, label in all_titles[:20]:
            if len(records) >= max_results:
                break
            params = {
                "action": "query",
                "titles": title,
                "prop": "links",
                "pllimit": 20,
                "format": "json",
            }
            try:
                r = requests.get(link_api, params=params, headers=HEADERS, timeout=15)
                r.raise_for_status()
                data = r.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    for link in page.get("links", []):
                        link_title = link.get("title", "")
                        if link_title in seen_titles:
                            continue
                        # Only fetch links that look relevant
                        keywords = ["law", "legal", "finance", "financial", "bank", "market",
                                    "regulation", "policy", "economy", "trade", "investment"]
                        if any(kw in link_title.lower() for kw in keywords):
                            rec = fetch_one(link_title, label)
                            if rec:
                                records.append(rec)
                                seen_titles.add(link_title)
                            if len(records) >= max_results:
                                break
                            time.sleep(0.3)
            except Exception:
                continue

    print(f"[Wikipedia] Fetched {len(records)} articles")
    return records


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def download_all(target_total: int = 1200) -> pd.DataFrame:
    """Fetch documents from all live APIs and merge into a single DataFrame."""
    all_records: List[Dict] = []

    print("=" * 60)
    print("Live API Document Downloader")
    print("=" * 60)

    # 1) ArXiv — scientific + financial (econ/q-fin)
    arxiv_records = fetch_arxiv(max_results=500)
    all_records.extend(arxiv_records)

    # 2) PubMed — medical/health
    pubmed_records = fetch_pubmed(max_results=450)
    all_records.extend(pubmed_records)

    # 3) Wikipedia — legal + financial
    wiki_records = fetch_wikipedia(max_results=350)
    all_records.extend(wiki_records)

    df = pd.DataFrame(all_records)

    # Deduplicate by doc_id
    df = df.drop_duplicates(subset=["doc_id"], keep="first")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    raw_path = RAW_DIR / "all_documents.jsonl"
    df.to_json(raw_path, orient="records", lines=True)
    print(f"\nSaved {len(df)} documents to {raw_path}")

    # Category distribution
    print("\nCategory distribution:")
    print(df["category"].value_counts())

    # Build 100-row sample CSV for README demos
    sample_df = df.head(100)[["doc_id", "source", "category", "title", "text", "date"]]
    sample_path = Path("data/sample/demo_100.csv")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved 100-row demo sample to {sample_path}")

    return df


if __name__ == "__main__":
    df = download_all(target_total=1200)
    print(f"\nDone. Total documents collected: {len(df)}")
