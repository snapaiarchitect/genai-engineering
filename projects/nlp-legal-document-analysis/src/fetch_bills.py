import requests
import json
import csv
import time
from datetime import datetime
import os

# Congress.gov API — free with DEMO_KEY for development
# https://api.congress.gov/v3/bill
# Real legislative documents: bills, resolutions, amendments

API_KEY = "DEMO_KEY"
BASE_URL = "https://api.congress.gov/v3/bill"

def fetch_bills(congress=118, limit=500):
    """Fetch real bills from Congress.gov API."""
    all_bills = []
    
    # Fetch multiple bill types for diversity
    bill_types = ["HR", "S", "HRES", "SRES", "HJRES", "SJRES", "HCONRES", "SCONRES"]
    per_type = limit // len(bill_types)
    
    print(f"Fetching bills from Congress {congress}")
    
    for bill_type in bill_types:
        offset = 0
        type_bills = []
        
        while len(type_bills) < per_type:
            params = {
                "api_key": API_KEY,
                "format": "json",
                "type": bill_type,
                "limit": min(250, per_type - len(type_bills)),
                "offset": offset,
            }
            
            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                bills = data.get("bills", [])
                if not bills:
                    break
                
                for bill in bills:
                    b = {
                        "congress": bill.get("congress", ""),
                        "number": bill.get("number", ""),
                        "type": bill.get("type", ""),
                        "origin_chamber": bill.get("originChamber", ""),
                        "title": bill.get("title", ""),
                        "latest_action_date": bill.get("latestAction", {}).get("actionDate", ""),
                        "latest_action_text": bill.get("latestAction", {}).get("text", ""),
                        "update_date": bill.get("updateDate", ""),
                        "url": bill.get("url", ""),
                    }
                    type_bills.append(b)
                
                offset += len(bills)
                
                pagination = data.get("pagination", {})
                if not pagination.get("next"):
                    break
                    
                time.sleep(0.3)
                
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching {bill_type}: {e}")
                break
        
        all_bills.extend(type_bills)
        print(f"  {bill_type}: {len(type_bills)} bills (total: {len(all_bills)})")
    
    return all_bills[:limit]

def save_bills(bills, filepath=None):
    """Save bills to CSV."""
    if filepath is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(PROJECT_ROOT, "data", "congress_bills.csv")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not bills:
        print("No bills to save")
        return
    
    keys = bills[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(bills)
    
    print(f"Saved {len(bills)} bills to {filepath}")
    
    # Type breakdown
    type_counts = {}
    chamber_counts = {}
    for b in bills:
        t = b.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1
        c = b.get("origin_chamber", "UNKNOWN")
        chamber_counts[c] = chamber_counts.get(c, 0) + 1
    
    meta = {
        "fetch_date": datetime.now().isoformat(),
        "api": "Congress.gov API v3",
        "url": BASE_URL,
        "api_key": API_KEY,
        "total_records": len(bills),
        "congress": bills[0].get("congress", "") if bills else "",
        "bill_types": type_counts,
        "chambers": chamber_counts,
    }
    
    with open(os.path.join(os.path.dirname(filepath), "bill_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved metadata")

if __name__ == "__main__":
    bills = fetch_bills(congress=118, limit=500)
    save_bills(bills)
    print(f"\nFetch complete. Total bills: {len(bills)}")
