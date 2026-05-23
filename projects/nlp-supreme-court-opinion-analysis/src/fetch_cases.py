import requests
import json
import csv
import time
from datetime import datetime
import os

# Oyez API — Supreme Court case data with justice voting records
# https://api.oyez.org/
# Focus: 2022 term cases with complete decisions (different from scotus-opinions text NLP)

BASE_URL = "https://api.oyez.org/cases"

def fetch_cases_with_decisions(term=2022, max_cases=100):
    """Fetch SCOTUS cases with decision data from Oyez API."""
    print(f"Fetching {term} term cases from Oyez API...")
    
    # Get case list
    params = {"filter": f"term:{term}", "per_page": max_cases}
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        cases = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching case list: {e}")
        return []
    
    print(f"Found {len(cases)} cases in term {term}")
    
    detailed_cases = []
    
    for case in cases[:max_cases]:
        case_url = case.get("href", "")
        if not case_url:
            continue
        
        try:
            detail = requests.get(case_url, headers=headers, timeout=30).json()
            if isinstance(detail, list):
                detail = detail[0] if detail else {}
            
            # Extract decision data
            decisions = detail.get("decisions", [])
            timeline = detail.get("timeline", [])
            
            if not decisions:
                continue
            
            case_data = {
                "id": case.get("ID", ""),
                "name": case.get("name", ""),
                "docket_number": case.get("docket_number", ""),
                "term": case.get("term", term),
                "question": case.get("question", ""),
                "description": case.get("description", ""),
                "justia_url": case.get("justia_url", ""),
            }
            
            # Timeline events
            for event in timeline:
                event_name = event.get("event", "").lower()
                dates = event.get("dates", [])
                if dates:
                    timestamp = datetime.fromtimestamp(dates[0]).isoformat()
                    if "granted" in event_name:
                        case_data["date_granted"] = timestamp
                    elif "argued" in event_name:
                        case_data["date_argued"] = timestamp
                    elif "decided" in event_name:
                        case_data["date_decided"] = timestamp
            
            # Decision data
            if decisions:
                decision = decisions[0]
                case_data["majority_votes"] = decision.get("majority_vote", 0)
                case_data["minority_votes"] = decision.get("minority_vote", 0)
                case_data["winning_party"] = decision.get("winning_party", "")
                case_data["decision_type"] = decision.get("decision_type", "")
                
                # Justice votes
                votes = decision.get("votes", []) or []
                for vote in votes:
                    member = vote.get("member", {})
                    if not member:
                        continue
                    justice = member.get("name", "").replace(" ", "_")
                    vote_type = vote.get("vote", "")
                    opinion_type = vote.get("opinion_type", "")
                    if justice:
                        case_data[f"vote_{justice}"] = vote_type
                        case_data[f"opinion_{justice}"] = opinion_type
            
            detailed_cases.append(case_data)
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching case detail: {e}")
            continue
        
        time.sleep(0.3)
    
    return detailed_cases

def save_cases(cases, filepath=None):
    """Save cases to CSV."""
    if filepath is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(PROJECT_ROOT, "data", "scotus_cases.csv")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not cases:
        print("No cases to save")
        return
    
    # Collect all unique keys
    all_keys = set()
    for case in cases:
        all_keys.update(case.keys())
    
    keys = sorted(all_keys, key=lambda k: (
        0 if k in ["id", "name", "docket_number", "term"] else
        1 if k.startswith("date_") else
        2 if k in ["majority_votes", "minority_votes", "winning_party", "decision_type"] else
        3 if k.startswith("vote_") else
        4 if k.startswith("opinion_") else
        5
    ))
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)
    
    print(f"Saved {len(cases)} cases to {filepath}")
    
    # Metadata
    justices = set()
    for case in cases:
        for key in case:
            if key.startswith("vote_"):
                justices.add(key.replace("vote_", "").replace("_", " "))
    
    meta = {
        "fetch_date": datetime.now().isoformat(),
        "api": "Oyez API",
        "url": "https://api.oyez.org/",
        "term": cases[0].get("term", "") if cases else "",
        "total_cases": len(cases),
        "justices_observed": sorted(list(justices)),
        "date_range": {
            "earliest_argued": min([c.get("date_argued", "") for c in cases if c.get("date_argued")], default=""),
            "latest_decided": max([c.get("date_decided", "") for c in cases if c.get("date_decided")], default=""),
        }
    }
    
    with open(os.path.join(os.path.dirname(filepath), "case_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print("Saved metadata")

if __name__ == "__main__":
    cases = fetch_cases_with_decisions(term=2022, max_cases=100)
    save_cases(cases)
    print(f"\nFetch complete. Total cases with decisions: {len(cases)}")
