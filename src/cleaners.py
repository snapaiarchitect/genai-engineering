"""
Data cleaning and transformation utilities.
Extracted from notebooks into reusable production modules.
"""
import re
import html
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("genai_cleaners")


def clean_text(text: str, remove_urls: bool = True, remove_emails: bool = True) -> str:
    """
    Clean raw text: strip HTML entities, normalize whitespace,
    optionally remove URLs and email addresses.
    """
    if not text or not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    if remove_urls:
        text = re.sub(r"https?://\S+", "", text)

    # Remove emails
    if remove_emails:
        text = re.sub(r"\S+@\S+", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_abstract(abstract: str) -> str:
    """Specialized cleaner for academic abstracts."""
    if not abstract:
        return ""
    abstract = clean_text(abstract, remove_urls=True, remove_emails=False)
    # Remove "Background:", "Methods:", "Results:", "Conclusion:" headers
    abstract = re.sub(r"\b(Background|Objective|Methods?|Results?|Conclusions?|Aim|Purpose):\s*", " ", abstract, flags=re.IGNORECASE)
    return abstract.strip()


def normalize_column_names(df_columns: List[str]) -> List[str]:
    """
    Convert DataFrame column names to snake_case.
    """
    cleaned = []
    for col in df_columns:
        # Lowercase, replace spaces/special chars with underscore
        col = col.lower().strip()
        col = re.sub(r"[^a-z0-9]+", "_", col)
        col = col.strip("_")
        cleaned.append(col)
    return cleaned


def validate_nct_id(nct_id: str) -> bool:
    """Validate ClinicalTrials.gov NCT ID format (NCT + 8 digits)."""
    return bool(re.match(r"^NCT\d{8}$", nct_id))


def validate_doi(doi: str) -> bool:
    """Validate DOI format."""
    return bool(re.match(r"^10\.\d{4,}/.+", doi))


def deduplicate_records(records: List[Dict], key_field: str = "id") -> List[Dict]:
    """Remove duplicate records by key field, keeping first occurrence."""
    seen = set()
    unique = []
    for r in records:
        key = r.get(key_field)
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def extract_year_from_date(date_str: str) -> Optional[int]:
    """Extract 4-digit year from ISO date string."""
    if not date_str:
        return None
    match = re.search(r"(\d{4})", date_str)
    return int(match.group(1)) if match else None


def safe_numeric(value: Any, default: float = 0.0) -> float:
    """Coerce value to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
