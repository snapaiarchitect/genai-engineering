"""
Source: arXiv API (export.arxiv.org/api/query)
License: arXiv.org perpetual, nonexclusive license
Fetches 2,000+ real abstracts across ML, NLP, CV, and Data Science.
"""
import requests
import json
import time
import os
from xml.etree import ElementTree as ET

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
QUERIES = [
    "machine learning", "natural language processing", "computer vision", "data science",
    "deep learning", "reinforcement learning", "neural networks", "large language models",
]
CAT_QUERIES = ["cat:cs.LG", "cat:cs.CL", "cat:cs.CV", "cat:cs.AI", "cat:stat.ML"]
ALL_QUERIES = QUERIES + CAT_QUERIES
BATCH = 100
MAX_PER_QUERY = 500


def fetch_query(query, max_results):
    records = []
    for start in range(0, max_results, BATCH):
        if query.startswith("cat:"):
            sq = query
        else:
            sq = f"all:{query.replace(' ', '+')}"
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={sq}&"
            f"start={start}&max_results={BATCH}&sortBy=submittedDate&sortOrder=descending"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for entry in root.findall("atom:entry", ARXIV_NS):
            id_url = entry.find("atom:id", ARXIV_NS).text
            arxiv_id = id_url.split("/abs/")[-1]
            record = {
                "id": arxiv_id,
                "title": entry.find("atom:title", ARXIV_NS).text.strip().replace("\n", " "),
                "authors": [a.find("atom:name", ARXIV_NS).text for a in entry.findall("atom:author", ARXIV_NS)],
                "summary": entry.find("atom:summary", ARXIV_NS).text.strip().replace("\n", " "),
                "categories": [c.get("term") for c in entry.findall("atom:category", ARXIV_NS)],
                "published": entry.find("atom:published", ARXIV_NS).text,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
            records.append(record)
        time.sleep(4)
    return records


def main():
    os.makedirs("data/raw", exist_ok=True)
    all_records = []
    seen_ids = set()
    for q in ALL_QUERIES:
        print(f"Fetching: {q} ...")
        batch = fetch_query(q, MAX_PER_QUERY)
        for r in batch:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_records.append(r)
        print(f"  → {len(batch)} fetched, total unique {len(all_records)}")
    with open("data/raw/arxiv_corpus.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_records)} abstracts to data/raw/arxiv_corpus.json")


if __name__ == "__main__":
    main()
