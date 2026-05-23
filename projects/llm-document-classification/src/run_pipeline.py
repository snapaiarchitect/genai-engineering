"""
End-to-End Pipeline Orchestrator
================================
Runs the complete document classification pipeline:
  1. Download documents (if not present)
  2. Clean text
  3. Engineer features
  4. Train models
  5. Evaluate and report metrics

Source data: ArXiv, PubMed, Wikipedia live API corpus.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from download_documents import download_all
from clean_data import clean_documents
from feature_engineering import build_features
from train_classifier import run_training
from evaluate_model import evaluate_all


def run_pipeline(force_download: bool = False):
    """Execute the full pipeline."""
    raw_path = Path("data/raw/all_documents.jsonl")
    if force_download or not raw_path.exists():
        print("[pipeline] Step 1/5: Downloading documents...")
        download_all()
    else:
        print("[pipeline] Step 1/5: Documents already present — skipping download")

    print("[pipeline] Step 2/5: Cleaning text...")
    clean_documents()

    print("[pipeline] Step 3/5: Engineering features...")
    build_features()

    print("[pipeline] Step 4/5: Training models...")
    run_training()

    print("[pipeline] Step 5/5: Evaluating models...")
    evaluate_all()

    print("[pipeline] ✅ Pipeline complete. Artifacts in models/ and reports/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full document classification pipeline")
    parser.add_argument("--force-download", action="store_true", help="Re-download documents")
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download)
