import requests
import json
import csv
import time
from datetime import datetime
import os

# CourtListener API — free legal research database
# https://www.courtlistener.com/api/rest-info/
# Focus: U.S. Courts of Appeals (circuit courts) — different from SCOTUS

def fetch_circuit_opinions(court_prefix="ca", limit=500):
    """Fetch real circuit court opinions from CourtListener API."""
    base_url = "https://www.courtlistener.com/api/rest/v3/opinions/"
    
    # Court codes: ca1, ca2, ca3, ca4, ca5, ca6, ca7, ca8, ca9, ca10, ca11, cadc
    circuit_courts = [f"ca{i}" for i in range(1, 12)] + ["cadc", "cafc"]
    
    all_opinions = []
    
    for court in circuit_courts:
        if len(all_opinions) >= limit:
            break
            
        params = {
            "court": court,
            "type": "010combined",  # Published opinions
            "order_by": "-date_filed",
            "page_size": 50,
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                continue
                
            for op in results:
                opinion = {
                    "id": op.get("id", ""),
                    "case_name": op.get("case_name", ""),
                    "court": court.upper(),
                    "circuit": court.replace("ca", "").replace("cadc", "DC").replace("cafc", "FC"),
                    "date_filed": op.get("date_filed", ""),
                    "status": op.get("status", ""),
                    "precedential": op.get("precedential_status", ""),
                    "citation_count": op.get("citation_count", 0),
                    "docket_number": op.get("docket_number", ""),
                    "snippet": op.get("snippet", "").replace("\n", " ")[:500] if op.get("snippet") else "",
                    "word_count": len(op.get("plain_text", "").split()) if op.get("plain_text") else 0,
                    "download_url": op.get("download_url", ""),
                    "absolute_url": op.get("absolute_url", ""),
                }
                all_opinions.append(opinion)
            
            print(f"  {court.upper()}: {len(results)} opinions (total: {len(all_opinions)})")
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {court}: {e}")
            continue
    
    return all_opinions[:limit]

def save_opinions(opinions, filepath=None):
    """Save opinions to CSV."""
    if filepath is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(PROJECT_ROOT, "data", "circuit_opinions.csv")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not opinions:
        print("No opinions to save")
        return
    
    keys = opinions[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(opinions)
    
    print(f"Saved {len(opinions)} opinions to {filepath}")
    
    meta = {
        "fetch_date": datetime.now().isoformat(),
        "api": "CourtListener API v3",
        "url": "https://www.courtlistener.com/api/rest/v3/opinions/",
        "total_records": len(opinions),
        "courts": list(set([o["court"] for o in opinions])),
        "circuits": list(set([o["circuit"] for o in opinions])),
        "date_range": {
            "earliest": min([o["date_filed"] for o in opinions if o["date_filed"]]),
            "latest": max([o["date_filed"] for o in opinions if o["date_filed"]]),
        }
    }
    
    with open(os.path.join(os.path.dirname(filepath), "opinion_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved metadata")

if __name__ == "__main__":
    opinions = fetch_circuit_opinions(limit=500)
    save_opinions(opinions)
    print(f"\nFetch complete. Total opinions: {len(opinions)}")
