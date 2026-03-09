import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from rag import answer_query

BENCHMARK_QUERIES = [
    {"query": "How do I apply for extenuating circumstances?", "expected_topic": "extenuating circumstances"},
    {"query": "What happens if I miss an exam?", "expected_topic": "missed exam"},
    {"query": "How do I change my programme or module?", "expected_topic": "programme change"},
    {"query": "Where can I find wellbeing support?", "expected_topic": "wellbeing"},
    {"query": "How do I get a replacement student ID card?", "expected_topic": "student ID"},
    {"query": "What is the deadline for tuition fee payment?", "expected_topic": "tuition fees"},
    {"query": "How do I request a transcript?", "expected_topic": "transcript"},
    {"query": "Can I get an extension for my coursework?", "expected_topic": "extension"},
    {"query": "How do I enrol for the new academic year?", "expected_topic": "enrolment"},
    {"query": "What is the capital of France?", "expected_topic": "__out_of_domain__"},
]


def run_evaluation():
    print(f"\n{'='*60}")
    print(f"KCL RAG Chatbot — Evaluation Run ({datetime.now().isoformat()})")
    print(f"{'='*60}\n")

    results = []
    for i, bq in enumerate(BENCHMARK_QUERIES, 1):
        query = bq["query"]
        expected = bq["expected_topic"]
        print(f"[{i}/{len(BENCHMARK_QUERIES)}] {query}")

        start = time.time()
        result = answer_query(query)
        elapsed = time.time() - start

        scores = result["retrieval_scores"]
        top_score = scores[0] if scores else 0.0
        has_context = len(scores) > 0

        entry = {
            "query": query, "expected_topic": expected,
            "top_score": round(top_score, 4),
            "num_sources": len(result["sources"]),
            "has_context": has_context,
            "answer_length": len(result["answer"]),
            "latency_s": round(elapsed, 2),
        }
        results.append(entry)
        print(f"  top={top_score:.3f}  sources={len(result['sources'])}  latency={elapsed:.1f}s\n")

    # Summary
    in_domain = [r for r in results if r["expected_topic"] != "__out_of_domain__"]
    avg_top = sum(r["top_score"] for r in in_domain) / len(in_domain) if in_domain else 0
    coverage = sum(1 for r in in_domain if r["has_context"]) / len(in_domain) * 100 if in_domain else 0

    print(f"{'─'*60}")
    print(f"  Avg top-1 score : {avg_top:.3f}")
    print(f"  Coverage        : {coverage:.1f}%")
    print(f"{'─'*60}\n")

    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Results saved to data/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()