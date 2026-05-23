import requests
import json
import csv
import time
from datetime import datetime
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ClinicalTrials.gov API v2 — free, no key required
# https://clinicaltrials.gov/api/v2/studies

def fetch_trials(condition="cancer", limit=500):
    """Fetch real clinical trial data from ClinicalTrials.gov API."""
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond": condition,
        "pageSize": 100,
        "filter.overallStatus": "RECRUITING",  # Active trials only for cleaner analysis
    }
    
    all_trials = []
    next_token = None
    page = 0
    
    print(f"Fetching clinical trials for condition: {condition}")
    
    while len(all_trials) < limit:
        if next_token:
            params["pageToken"] = next_token
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            studies = data.get("studies", [])
            if not studies:
                break
                
            for study in studies:
                protocol = study.get("protocolSection", {})
                
                trial = {
                    "nct_id": protocol.get("identificationModule", {}).get("nctId", ""),
                    "title": protocol.get("identificationModule", {}).get("briefTitle", ""),
                    "condition": ", ".join(protocol.get("conditionsModule", {}).get("conditions", [])),
                    "phase": ", ".join(protocol.get("designModule", {}).get("phases", [])),
                    "overall_status": protocol.get("statusModule", {}).get("overallStatus", ""),
                    "enrollment_count": protocol.get("designModule", {}).get("enrollmentInfo", {}).get("count", 0),
                    "lead_sponsor": protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
                    "sponsor_class": protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("class", ""),
                    "start_date": protocol.get("statusModule", {}).get("startDateStruct", {}).get("date", ""),
                    "completion_date": protocol.get("statusModule", {}).get("completionDateStruct", {}).get("date", ""),
                    "study_type": protocol.get("designModule", {}).get("studyType", ""),
                    "intervention_type": ", ".join([i.get("type", "") for i in protocol.get("armsInterventionsModule", {}).get("interventions", [])]),
                    "locations": len(protocol.get("contactsLocationsModule", {}).get("locations", [])),
                }
                all_trials.append(trial)
            
            next_token = data.get("nextPageToken")
            page += 1
            print(f"  Page {page}: {len(studies)} trials (total: {len(all_trials)})")
            
            if not next_token:
                break
                
            time.sleep(0.5)  # Respect rate limits
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
    
    return all_trials[:limit]

def save_trials(trials, filepath=None):
    """Save trials to CSV."""
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data", "clinical_trials.csv")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not trials:
        print("No trials to save")
        return
    
    keys = trials[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trials)
    
    print(f"Saved {len(trials)} trials to {filepath}")
    
    # Save metadata
    meta = {
        "fetch_date": datetime.now().isoformat(),
        "api": "ClinicalTrials.gov API v2",
        "url": "https://clinicaltrials.gov/api/v2/studies",
        "total_records": len(trials),
        "conditions": list(set([t["condition"] for t in trials if t["condition"]])),
        "phases": list(set([t["phase"] for t in trials if t["phase"]])),
    }
    
    with open(os.path.join(PROJECT_ROOT, "data", "trial_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved metadata to data/trial_metadata.json")

if __name__ == "__main__":
    trials = fetch_trials(condition="cancer", limit=500)
    save_trials(trials)
    print(f"\nFetch complete. Total trials: {len(trials)}")
