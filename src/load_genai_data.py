import pandas as pd
import requests

"""
GenAI / NLP data loaders — real public document corpora.
"""

def load_scotus_opinions(court='supreme', start_year=2010, end_year=2024):
    """
    CourtListener — SCOTUS opinions.
    API: https://www.courtlistener.com/api/rest/v3/
    Data: 30K+ Supreme Court opinions with full text
    Access: Free API (rate limited)
    """
    base = "https://www.courtlistener.com/api/rest/v3/"
    # Endpoint: opinions/?court=scotus&date_filed__gte=2010-01-01
    pass

def load_arxiv_papers(category='cs.AI', max_results=1000):
    """
    arXiv API — Research papers.
    API: http://export.arxiv.org/api/query
    Data: 2M+ papers with titles, abstracts, authors
    Access: Free, rate limited to 1 query/3 seconds
    """
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    pass

def load_pubmed_articles(query="machine learning", max_results=1000):
    """
    PubMed / NCBI E-utilities.
    API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
    Data: 35M+ biomedical citations
    Access: Free (3 requests/second without API key)
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    # esearch.fcgi?db=pubmed&term=machine+learning&retmax=1000
    pass
