"""
Source: arXiv API corpus (data/raw/arxiv_corpus.json)
Embeds abstracts with sentence-transformers all-MiniLM-L6-v2.
Builds FAISS L2 index + metadata mapping.
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/raw/arxiv_corpus.json"
INDEX_DIR = "data/vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} abstracts")

    texts = [r["summary"] for r in corpus]
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))

    metadata = {
        i: {
            "id": r["id"],
            "title": r["title"],
            "authors": r["authors"],
            "categories": r["categories"],
            "published": r["published"],
            "url": r["url"],
            "summary": r["summary"],
        }
        for i, r in enumerate(corpus)
    }
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Index: {index.ntotal} vectors, dim={dim}")
    print(f"Saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
