import requests
import json
import csv
import os

# U.S. Census ACS API — American Community Survey 5-Year Estimates
# https://www.census.gov/data/developers/data-sets/acs-5year.html
# Real demographic data for model training

# Get API key from environment variable — set via GitHub Secrets or local .env
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")
if not CENSUS_API_KEY:
    raise ValueError("CENSUS_API_KEY environment variable not set. Get one at https://api.census.gov/data/key_signup.html")
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

def fetch_acs_data():
    """Fetch real Census ACS data for all US counties."""
    # Variables: median income, population, education, employment
    variables = [
        "NAME",                           # County name
        "B19013_001E",                    # Median household income
        "B01003_001E",                    # Total population
        "B15003_022E",                    # Bachelor's degree holders
        "B23027_002E",                    # In labor force
        "B25064_001E",                    # Median gross rent
        "B08303_001E",                    # Mean travel time to work
        "B25003_002E",                    # Owner-occupied housing
        "B25003_003E",                    # Renter-occupied housing
    ]
    
    params = {
        "get": ",".join(variables),
        "for": "county:*",
        "in": "state:*",
        "key": CENSUS_API_KEY,
    }
    
    print("Fetching Census ACS 2022 data for all US counties...")
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # First row is headers
        headers = data[0]
        rows = data[1:]
        
        records = []
        for row in rows:
            record = dict(zip(headers, row))
            # Convert numeric fields
            numeric_fields = ["B19013_001E", "B01003_001E", "B15003_022E", "B23027_002E", 
                            "B25064_001E", "B08303_001E", "B25003_002E", "B25003_003E"]
            for field in numeric_fields:
                try:
                    record[field] = float(record.get(field, 0))
                except (ValueError, TypeError):
                    record[field] = 0.0
            records.append(record)
        
        print(f"Fetched {len(records)} counties")
        return records, headers
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return [], []

def save_data(records, headers):
    """Save ACS data to CSV."""
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(PROJECT_ROOT, "data", "acs_county_data.csv")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Rename columns for readability
    rename_map = {
        "NAME": "county_name",
        "B19013_001E": "median_income",
        "B01003_001E": "population",
        "B15003_022E": "bachelors_degree",
        "B23027_002E": "labor_force",
        "B25064_001E": "median_rent",
        "B08303_001E": "commute_time",
        "B25003_002E": "owner_occupied",
        "B25003_003E": "renter_occupied",
        "state": "state_fips",
        "county": "county_fips",
    }
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[rename_map.get(h, h) for h in headers])
        writer.writeheader()
        for record in records:
            renamed = {rename_map.get(k, k): v for k, v in record.items()}
            writer.writerow(renamed)
    
    print(f"Saved {len(records)} counties to {filepath}")
    
    # Metadata
    meta = {
        "fetch_date": "2026-05-21",
        "api": "U.S. Census ACS 5-Year API",
        "url": BASE_URL,
        "year": 2022,
        "total_records": len(records),
        "variables": {
            "median_income": "B19013_001E — Median household income",
            "population": "B01003_001E — Total population",
            "bachelors_degree": "B15003_022E — Bachelor's degree or higher",
            "labor_force": "B23027_002E — In labor force",
            "median_rent": "B25064_001E — Median gross rent",
            "commute_time": "B08303_001E — Mean travel time to work",
            "owner_occupied": "B25003_002E — Owner-occupied housing units",
            "renter_occupied": "B25003_003E — Renter-occupied housing units",
        }
    }
    
    with open(os.path.join(os.path.dirname(filepath), "acs_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print("Saved metadata")

if __name__ == "__main__":
    records, headers = fetch_acs_data()
    if records:
        save_data(records, headers)
        print(f"\nFetch complete. Total counties: {len(records)}")
    else:
        print("\nFetch failed.")
