"""
Source: FAISS index (data/vector_store/faiss.index) + metadata
Query transformation with synonym expansion and HyDE-style pseudo-doc generation.
Dense retrieval via FAISS L2 search.
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/vector_store/faiss.index"
META_PATH = "data/vector_store/metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

SYNONYMS = {
    "attention": ["attention mechanism", "self-attention", "transformer attention"],
    "transformer": ["transformer model", "attention-based model", "BERT", "GPT"],
    "classification": ["classifier", "categorization", "label prediction"],
    "clustering": ["cluster analysis", "k-means", "unsupervised grouping"],
    "optimization": ["gradient descent", "SGD", "Adam optimizer", "learning rate"],
}


def expand_query(query):
    tokens = query.lower().split()
    expanded = [query]
    for t in tokens:
        if t in SYNONYMS:
            expanded.extend(SYNONYMS[t])
    return " ".join(expanded)


def hyde_pseudo_doc(query, model):
    return f"A research paper about {query}. This paper discusses methods, experiments, and results."


def retrieve(query, k=10, use_hyde=True):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)

    q_expanded = expand_query(query)
    if use_hyde:
        q_expanded = hyde_pseudo_doc(q_expanded, model)

    q_emb = model.encode([q_expanded])
    q_emb = np.array(q_emb).astype("float32")
    distances, indices = index.search(q_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = metadata.get(str(idx), metadata.get(int(idx), {}))
        results.append({
            "index": int(idx),
            "score": float(dist),
            **meta,
        })
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    res = retrieve(args.query, k=args.k)
    for i, r in enumerate(res, 1):
        print(f"\n{i}. {r['title']}\n   Score: {r['score']:.4f} | {r['url']}\n   {r['summary'][:200]}...")


if __name__ == "__main__":
    main()
