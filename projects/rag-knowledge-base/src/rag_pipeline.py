"""
Source: arXiv API corpus + FAISS index + cross-encoder reranker
End-to-end RAG pipeline: query → transform → retrieve → rerank → generate.
CLI: python src/rag_pipeline.py --query "..." --k 5
"""
import argparse
import sys
sys.path.insert(0, "src")
from retriever import retrieve
from reranker import rerank
from generator import generate_answer


def run(query, k=5):
    print(f"Query: {query}\n")
    print("Retrieving...")
    docs = retrieve(query, k=max(k * 4, 20))
    print(f"  {len(docs)} candidates")
    print("Reranking...")
    ranked = rerank(query, docs, top_k=k)
    print(f"  top-{k} selected")
    print("=" * 60)
    answer = generate_answer(query, ranked)
    print(answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--k", type=int, default=5, help="Number of documents")
    args = parser.parse_args()
    run(args.query, k=args.k)


if __name__ == "__main__":
    main()
