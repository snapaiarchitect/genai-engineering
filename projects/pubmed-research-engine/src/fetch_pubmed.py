import requests
import xml.etree.ElementTree as ET
import csv
import json
import time
from datetime import datetime
import os

# PubMed E-utilities API — NCBI search and retrieval
# https://www.ncbi.nlm.nih.gov/books/NBK25500/
# No API key required for basic access (rate limit: 3 requests/second)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def search_pubmed(query="machine learning", max_results=500):
    """Search PubMed and fetch article metadata."""
    print(f"Searching PubMed for: '{query}'")
    
    # Step 1: Search for IDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, 10000),
        "retmode": "json",
        "sort": "date",
    }
    
    try:
        r = requests.get(f"{BASE_URL}/esearch.fcgi", params=search_params, timeout=30)
        r.raise_for_status()
        data = r.json()
        idlist = data.get("esearchresult", {}).get("idlist", [])
        print(f"  Found {len(idlist)} articles")
    except requests.exceptions.RequestException as e:
        print(f"Error searching: {e}")
        return []
    
    if not idlist:
        return []
    
    # Step 2: Fetch summaries in batches
    all_articles = []
    batch_size = 200
    
    for i in range(0, len(idlist), batch_size):
        batch = idlist[i:i+batch_size]
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        
        try:
            r = requests.get(f"{BASE_URL}/efetch.fcgi", params=fetch_params, timeout=60)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            
            for article in root.findall(".//PubmedArticle"):
                medline = article.find(".//MedlineCitation")
                if medline is None:
                    continue
                
                pmid = medline.find("PMID")
                pmid = pmid.text if pmid is not None else ""
                
                title = medline.find(".//ArticleTitle")
                title = title.text if title is not None else ""
                
                abstract_text = ""
                abstract = medline.find(".//Abstract")
                if abstract is not None:
                    for ab_text in abstract.findall("AbstractText"):
                        if ab_text.text:
                            abstract_text += ab_text.text + " "
                
                journal = medline.find(".//Journal/Title")
                journal = journal.text if journal is not None else ""
                
                year = medline.find(".//DateCompleted/Year") or medline.find(".//Article/Journal/JournalIssue/PubDate/Year")
                year = year.text if year is not None else ""
                
                month = medline.find(".//DateCompleted/Month") or medline.find(".//Article/Journal/JournalIssue/PubDate/Month")
                month = month.text if month is not None else ""
                
                authors = []
                author_list = medline.find(".//AuthorList")
                if author_list is not None:
                    for author in author_list.findall("Author"):
                        last = author.find("LastName")
                        first = author.find("ForeName")
                        if last is not None and last.text:
                            name = last.text
                            if first is not None and first.text:
                                name = f"{first.text} {last.text}"
                            authors.append(name)
                
                mesh_terms = []
                mesh_list = medline.find(".//MeshHeadingList")
                if mesh_list is not None:
                    for heading in mesh_list.findall("MeshHeading"):
                        descriptor = heading.find("DescriptorName")
                        if descriptor is not None and descriptor.text:
                            mesh_terms.append(descriptor.text)
                
                article_data = {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract_text.strip(),
                    "journal": journal,
                    "year": year,
                    "month": month,
                    "authors": "; ".join(authors[:5]),  # Top 5 authors
                    "mesh_terms": "; ".join(mesh_terms[:10]),  # Top 10 MeSH terms
                    "author_count": len(authors),
                    "mesh_count": len(mesh_terms),
                }
                all_articles.append(article_data)
            
            print(f"  Fetched batch {i//batch_size + 1}: {len(batch)} articles (total: {len(all_articles)})")
            time.sleep(0.4)  # Rate limit compliance
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching batch: {e}")
            continue
    
    return all_articles

def save_articles(articles, filepath=None):
    """Save articles to CSV."""
    if filepath is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(PROJECT_ROOT, "data", "pubmed_abstracts.csv")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not articles:
        print("No articles to save")
        return
    
    keys = articles[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(articles)
    
    print(f"Saved {len(articles)} articles to {filepath}")
    
    # Journal breakdown
    journals = {}
    years = {}
    for art in articles:
        j = art.get("journal", "Unknown")
        journals[j] = journals.get(j, 0) + 1
        y = art.get("year", "Unknown")
        if y:
            years[y] = years.get(y, 0) + 1
    
    meta = {
        "fetch_date": datetime.now().isoformat(),
        "api": "NCBI E-utilities (PubMed)",
        "url": BASE_URL,
        "query": "machine learning",
        "total_records": len(articles),
        "top_journals": sorted(journals.items(), key=lambda x: x[1], reverse=True)[:10],
        "year_distribution": dict(sorted(years.items(), key=lambda x: x[0])),
    }
    
    with open(os.path.join(os.path.dirname(filepath), "pubmed_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print("Saved metadata")

if __name__ == "__main__":
    articles = search_pubmed(query="machine learning", max_results=500)
    save_articles(articles)
    print(f"\nFetch complete. Total articles: {len(articles)}")
