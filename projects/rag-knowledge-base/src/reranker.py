"""
Source: FAISS retrieval results
Cross-encoder re-ranking using ms-marco-MiniLM-L-6-v2.
Returns top-k with confidence scores.
"""
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query, docs, top_k=5):
    model = CrossEncoder(MODEL_NAME)
    pairs = [(query, d["summary"]) for d in docs]
    scores = model.predict(pairs)
    scored = [(s, d) for s, d in zip(scores, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "rank": i + 1,
            "confidence": float(score),
            **doc,
        }
        for i, (score, doc) in enumerate(scored[:top_k])
    ]


def main():
    import json
    import argparse
    from retriever import retrieve
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    docs = retrieve(args.query, k=20)
    ranked = rerank(args.query, docs, top_k=args.k)
    for r in ranked:
        print(f"\nRank {r['rank']} | Confidence: {r['confidence']:.4f}\n{r['title']}\n{r['url']}")


if __name__ == "__main__":
    main()
