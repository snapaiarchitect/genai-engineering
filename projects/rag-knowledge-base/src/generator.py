"""
Source: Re-ranked retrieval results
Multi-document answer synthesis with source citations.
Uses keyword-based extractive summarization (no LLM API required).
"""
import re
from collections import Counter


def keyword_summarize(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop = {"this", "that", "with", "from", "they", "have", "been", "their", "than", "also", "using", "based", "such", "show", "used", "proposed", "paper", "model", "method", "results"}
    freqs = Counter(w for w in words if w not in stop)
    scored = []
    for s in sentences:
        score = sum(freqs.get(w, 0) for w in re.findall(r'\b[a-zA-Z]{4,}\b', s.lower()))
        scored.append((score, s))
    scored.sort(reverse=True)
    top = [s for _, s in scored[:max_sentences]]
    order = {s: i for i, s in enumerate(sentences)}
    top.sort(key=lambda s: order.get(s, 999))
    return " ".join(top)


def generate_answer(query, docs):
    if not docs:
        return "No relevant documents found."
    cited = []
    for i, d in enumerate(docs, 1):
        summary = keyword_summarize(d["summary"], max_sentences=2)
        authors = ", ".join(d["authors"][:2]) + (" et al." if len(d["authors"]) > 2 else "")
        cited.append({
            "num": i,
            "title": d["title"],
            "authors": authors,
            "url": d["url"],
            "summary": summary,
        })

    parts = [f"Based on [{c['num']}] {c['title']} ({c['authors']})" for c in cited]
    intro = " and ".join(parts) + ", the answer is:\n"
    body = "\n\n".join([f"[{c['num']}] {c['summary']}" for c in cited])
    refs = "\n\nSources:\n" + "\n".join([f"[{c['num']}] {c['title']} — {c['url']}" for c in cited])
    return intro + body + refs


def main():
    import argparse
    import json
    from retriever import retrieve
    from reranker import rerank
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    docs = retrieve(args.query, k=20)
    ranked = rerank(args.query, docs, top_k=args.k)
    print(generate_answer(args.query, ranked))


if __name__ == "__main__":
    main()
