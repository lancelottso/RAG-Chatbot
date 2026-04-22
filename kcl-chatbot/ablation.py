"""
ablation.py — Ablation study framework for the KCL RAG Chatbot.

Tests different configurations systematically and records results
for comparison. Supports testing:
    1. Chunk geometry (chunk size × overlap)
    2. Info volume (TOP_K × max_tokens)
    3. Query expansion (on vs off)
    4. Similarity threshold
    5. Heading bias (none vs first chunk vs all chunks)
    6. Temperature

Usage:
    python3 ablation.py --study chunk_geometry
    python3 ablation.py --study info_volume
    python3 ablation.py --study query_expansion
    python3 ablation.py --study similarity_threshold
    python3 ablation.py --study heading_bias
    python3 ablation.py --study temperature
    python3 ablation.py --study all
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(__file__))

from config import SCRAPED_PAGES_PATH, FAISS_INDEX_PATH, CHUNKS_PATH, DATA_DIR
from rag.chunker import chunk_pages, save_chunks
from rag.embedder import build_faiss_index, load_faiss_index, embed_texts, search, _expand_query
from rag.generator import generate_response

# Benchmark Queries (core set for ablation)
CORE_QUERIES = [
    {
        "query": "How do I apply for extenuating circumstances?",
        "reference_answer": "Log on to Student Records. Select Student Home Page. Select My Mitigating Circumstances within the My Modules area. Select New Request within the Submit a new Mitigating Circumstances request area.",
    },
    {
        "query": "What happens if I miss an exam?",
        "reference_answer": "If you miss an exam due to unforeseen circumstances, you should contact your department or faculty as soon as possible to understand the specific rules regarding missed assessments for your programme.",
    },
    {
        "query": "How do I change my programme or module?",
        "reference_answer": "Contact your department or faculty initially to discuss your options. Not all departments are able to permit changes to modules and many departments will have a deadline by which they can consider such changes.",
    },
    {
        "query": "Where can I find wellbeing support?",
        "reference_answer": "King's offers an online support service called Spectrum.Life that is free and confidential, available 24 hours a day, 7 days a week. Visit Spectrum.Life to register with your King's email and set Organisation code to KCL.",
    },
    {
        "query": "How do I get a replacement student ID card?",
        "reference_answer": "Report the loss immediately to Student Services or Security so it can be deactivated. Visit a service desk where they can produce a replacement while you wait. Bring photographic ID such as your passport.",
    },
    {
        "query": "What is the deadline for tuition fee payment?",
        "reference_answer": "Details on when you can pay your tuition fees can be found on the Credit Control webpages.",
    },
    {
        "query": "How do I request a transcript?",
        "reference_answer": "The transcript is issued only to students who have completed their course and been awarded. Once awarded, refer to information for graduates. If still studying, refer to information for current students.",
    },
    {
        "query": "Can I get an extension for my coursework?",
        "reference_answer": "Submit a Mitigating Circumstances Form. Extensions are not automatically granted. Log in to Student Records, select My Mitigating Circumstances, select New Request, choose Extension Granted by PAA from the dropdown.",
    },
    {
        "query": "How do I enrol for the new academic year?",
        "reference_answer": "You will receive an email inviting you to complete re-enrolment. Log into Student Records and navigate to My Enrolment page. Complete the task as soon as you can as access to library services and KEATS will be affected if late.",
    },
    {
        "query": "What is the capital of France?",
        "reference_answer": "",
    },
]


def _compare_answers(generated, reference):
    """Calculate recall between generated answer and reference."""
    if not reference:
        return {"recall": 1.0 if not generated.strip() else 0.0}

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "shall", "can",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "through", "during", "before", "after", "and",
                  "but", "or", "nor", "not", "so", "yet", "if", "then", "than",
                  "that", "this", "these", "those", "it", "its", "you", "your",
                  "i", "my", "we", "our", "they", "their", "he", "she", "his",
                  "her", "what", "which", "who", "whom", "how", "when", "where",
                  "why", "all", "each", "every", "both", "few", "more", "most",
                  "other", "some", "such", "no", "only", "own", "same", "about"}

    gen_words = set(generated.lower().split()) - stop_words
    ref_words = set(reference.lower().split()) - stop_words

    if not ref_words:
        return {"recall": 0.0}

    overlap = gen_words & ref_words
    recall = len(overlap) / len(ref_words)
    return {"recall": round(recall, 3)}


def _run_queries(queries, use_expansion=True):
    """Run a set of queries and return average metrics."""
    import rag.embedder as embedder

    results = []
    for bq in queries:
        query = bq["query"]
        ref = bq.get("reference_answer", "")

        if use_expansion:
            expanded = _expand_query(query)
            query_vec = embed_texts([expanded])
        else:
            query_vec = embed_texts([query])

        scores, indices = embedder._index.search(query_vec, 5)

        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < 0.35:
                continue
            chunk = embedder._chunks[idx]
            chunks.append({
                "text": chunk["text"],
                "source_url": chunk["source_url"],
                "page_title": chunk["page_title"],
                "score": float(score),
            })

        response = generate_response(query, chunks)
        similarity = _compare_answers(response["answer"], ref)

        top_score = chunks[0]["score"] if chunks else 0.0
        results.append({
            "query": query,
            "top_score": top_score,
            "recall": similarity["recall"],
            "has_context": len(chunks) > 0,
        })

    avg_retrieval = sum(r["top_score"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    coverage = sum(1 for r in results if r["has_context"]) / len(results)

    return {
        "avg_retrieval": round(avg_retrieval, 4),
        "avg_recall": round(avg_recall, 3),
        "coverage": round(coverage, 3),
    }


# Chunk Geometry
def study_chunk_geometry():
    """Test different chunk sizes and overlaps."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 1: Chunk Geometry")
    print("=" * 60)

    with open(SCRAPED_PAGES_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunk_sizes = [200, 300, 400, 500, 750]
    overlaps = [25, 50, 75, 100]

    results = []

    for size, overlap in product(chunk_sizes, overlaps):
        if overlap >= size:
            continue

        print(f"\n  Testing chunk_size={size}, overlap={overlap}...")

        chunks = chunk_pages(pages, chunk_size=size, chunk_overlap=overlap)
        save_chunks(chunks)
        build_faiss_index(chunks)

        import rag.embedder as embedder
        embedder._index = None
        embedder._chunks = None
        load_faiss_index()

        import rag.embedder as emb
        query_results = []
        for bq in CORE_QUERIES:
            query = bq["query"]
            expanded = _expand_query(query)
            query_vec = embed_texts([expanded])
            scores, indices = emb._index.search(query_vec, 5)

            top_score = float(scores[0][0]) if scores[0][0] != -1 else 0.0
            has_context = top_score >= 0.35
            query_results.append({"top_score": top_score, "has_context": has_context})

        avg_retrieval = sum(r["top_score"] for r in query_results) / len(query_results)
        coverage = sum(1 for r in query_results if r["has_context"]) / len(query_results)

        entry = {
            "chunk_size": size,
            "overlap": overlap,
            "num_chunks": len(chunks),
            "avg_retrieval": round(avg_retrieval, 4),
            "coverage": round(coverage, 3),
        }
        results.append(entry)
        print(f"    chunks={len(chunks)}  avg_score={avg_retrieval:.4f}  coverage={coverage:.1%}")

    out_path = "data/ablation_chunk_geometry.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    best = max(results, key=lambda r: r["avg_retrieval"])
    print(f"\nBest config: chunk_size={best['chunk_size']}, overlap={best['overlap']}, "
          f"avg_score={best['avg_retrieval']:.4f}")

    return results


# Info Volume
def study_info_volume():
    """Test different TOP_K and max_tokens values."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 2: Info Volume to LLM")
    print("=" * 60)

    import rag.embedder as embedder
    if embedder._index is None:
        load_faiss_index()

    top_k_values = [3, 5, 7, 10]
    max_token_values = [512, 1024]

    results = []

    for k, max_tok in product(top_k_values, max_token_values):
        print(f"\n  Testing top_k={k}, max_tokens={max_tok}...")

        query_results = []
        for bq in CORE_QUERIES:
            query = bq["query"]
            ref = bq.get("reference_answer", "")

            expanded = _expand_query(query)
            query_vec = embed_texts([expanded])
            scores, indices = embedder._index.search(query_vec, k)

            chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < 0.35:
                    continue
                chunk = embedder._chunks[idx]
                chunks.append({
                    "text": chunk["text"],
                    "source_url": chunk["source_url"],
                    "page_title": chunk["page_title"],
                    "score": float(score),
                })

            from huggingface_hub import InferenceClient
            from rag.generator import SYSTEM_PROMPT
            client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

            context_block = "\n\n---\n\n".join(
                f"[Source: {c['page_title']}]\n{c['text']}" for c in chunks
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ]

            try:
                response = client.chat_completion(
                    messages=messages,
                    model="HuggingFaceH4/zephyr-7b-beta",
                    max_tokens=max_tok,
                    temperature=0.3,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"    Error: {e}")
                answer = ""

            similarity = _compare_answers(answer, ref)
            top_score = chunks[0]["score"] if chunks else 0.0

            query_results.append({
                "top_score": top_score,
                "recall": similarity["recall"],
            })

            time.sleep(1)

        avg_retrieval = sum(r["top_score"] for r in query_results) / len(query_results)
        avg_recall = sum(r["recall"] for r in query_results) / len(query_results)

        entry = {
            "top_k": k,
            "max_tokens": max_tok,
            "avg_retrieval": round(avg_retrieval, 4),
            "avg_recall": round(avg_recall, 3),
        }
        results.append(entry)
        print(f"    avg_score={avg_retrieval:.4f}  avg_recall={avg_recall:.3f}")

    out_path = "data/ablation_info_volume.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    best = max(results, key=lambda r: r["avg_recall"])
    print(f"\nBest config: top_k={best['top_k']}, max_tokens={best['max_tokens']}, "
          f"recall={best['avg_recall']:.3f}")

    return results


# Query Expansion
def study_query_expansion():
    """Test query expansion on vs off."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 3: Query Expansion")
    print("=" * 60)

    import rag.embedder as embedder
    if embedder._index is None:
        load_faiss_index()

    results = []

    for use_expansion in [False, True]:
        label = "WITH expansion" if use_expansion else "WITHOUT expansion"
        print(f"\n  Testing {label}...")

        query_results = []
        for bq in CORE_QUERIES:
            query = bq["query"]

            if use_expansion:
                search_query = _expand_query(query)
            else:
                search_query = query

            query_vec = embed_texts([search_query])
            scores, indices = embedder._index.search(query_vec, 5)

            top_score = float(scores[0][0]) if scores[0][0] != -1 else 0.0
            has_context = top_score >= 0.35
            query_results.append({"top_score": top_score, "has_context": has_context})

        avg_retrieval = sum(r["top_score"] for r in query_results) / len(query_results)
        coverage = sum(1 for r in query_results if r["has_context"]) / len(query_results)

        entry = {
            "expansion": use_expansion,
            "avg_retrieval": round(avg_retrieval, 4),
            "coverage": round(coverage, 3),
        }
        results.append(entry)
        print(f"    avg_score={avg_retrieval:.4f}  coverage={coverage:.1%}")

    out_path = "data/ablation_query_expansion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    return results


# Similarity Threshold
def study_similarity_threshold():
    """Test different similarity thresholds for filtering results."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 4: Similarity Threshold")
    print("=" * 60)

    import rag.embedder as embedder
    if embedder._index is None:
        load_faiss_index()

    thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]

    results = []

    for threshold in thresholds:
        print(f"\n  Testing threshold={threshold}...")

        query_results = []
        for bq in CORE_QUERIES:
            query = bq["query"]
            ref = bq.get("reference_answer", "")
            is_ood = ref == ""

            expanded = _expand_query(query)
            query_vec = embed_texts([expanded])
            scores, indices = embedder._index.search(query_vec, 5)

            chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < threshold:
                    continue
                chunk = embedder._chunks[idx]
                chunks.append({
                    "text": chunk["text"],
                    "source_url": chunk["source_url"],
                    "page_title": chunk["page_title"],
                    "score": float(score),
                })

            has_context = len(chunks) > 0
            top_score = chunks[0]["score"] if chunks else 0.0

            # For out-of-domain: no context = correct
            if is_ood:
                correct_rejection = not has_context
                query_results.append({
                    "top_score": top_score,
                    "has_context": has_context,
                    "correct_rejection": correct_rejection,
                    "is_ood": True,
                })
            else:
                query_results.append({
                    "top_score": top_score,
                    "has_context": has_context,
                    "is_ood": False,
                })

        in_domain = [r for r in query_results if not r["is_ood"]]
        ood = [r for r in query_results if r["is_ood"]]

        coverage = sum(1 for r in in_domain if r["has_context"]) / len(in_domain)
        avg_retrieval = sum(r["top_score"] for r in in_domain) / len(in_domain)
        ood_rejected = all(r["correct_rejection"] for r in ood) if ood else True

        entry = {
            "threshold": threshold,
            "coverage": round(coverage, 3),
            "avg_retrieval": round(avg_retrieval, 4),
            "ood_correctly_rejected": ood_rejected,
            "chunks_returned_avg": round(sum(1 for r in in_domain if r["has_context"]) / len(in_domain) * 5, 1),
        }
        results.append(entry)
        print(f"    coverage={coverage:.1%}  avg_score={avg_retrieval:.4f}  ood_rejected={ood_rejected}")

    out_path = "data/ablation_similarity_threshold.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    # Find best threshold that maintains OOD rejection
    valid = [r for r in results if r["ood_correctly_rejected"]]
    if valid:
        best = max(valid, key=lambda r: r["coverage"])
        print(f"\nBest threshold: {best['threshold']} (coverage={best['coverage']:.1%}, OOD rejected)")

    return results


# Heading Bias
def study_heading_bias():
    """Test whether prepending headings to chunks helps retrieval."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 5: Heading Bias")
    print("=" * 60)

    with open(SCRAPED_PAGES_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    from config import CHUNK_SIZE, CHUNK_OVERLAP

    modes = ["none", "first_chunk", "all_chunks"]
    results = []

    for mode in modes:
        print(f"\n  Testing heading_bias={mode}...")

        all_chunks = []
        chunk_id = 0

        for page in pages:
            body = page.get("body", "")
            if not body or len(body.split()) < 20:
                continue

            heading_context = ""
            if mode != "none":
                if page.get("title"):
                    heading_context += page["title"] + ". "
                if page.get("headings"):
                    heading_context += " — ".join(page["headings"][:5]) + ". "

            words = body.split()
            start = 0
            first = True
            while start < len(words):
                end = start + CHUNK_SIZE
                chunk_text = " ".join(words[start:end])
                if chunk_text.strip():
                    if mode == "all_chunks":
                        enriched = heading_context + chunk_text
                    elif mode == "first_chunk" and first:
                        enriched = heading_context + chunk_text
                    else:
                        enriched = chunk_text

                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "text": enriched,
                        "source_url": page.get("url", ""),
                        "page_title": page.get("title", ""),
                        "headings": page.get("headings", []),
                    })
                    chunk_id += 1
                    first = False
                start += CHUNK_SIZE - CHUNK_OVERLAP

        print(f"    Created {len(all_chunks)} chunks")

        # Save and index
        save_chunks(all_chunks)
        build_faiss_index(all_chunks)

        import rag.embedder as embedder
        embedder._index = None
        embedder._chunks = None
        load_faiss_index()

        query_results = []
        for bq in CORE_QUERIES:
            query = bq["query"]
            expanded = _expand_query(query)
            query_vec = embed_texts([expanded])
            scores, indices = embedder._index.search(query_vec, 5)

            top_score = float(scores[0][0]) if scores[0][0] != -1 else 0.0
            has_context = top_score >= 0.35
            query_results.append({"top_score": top_score, "has_context": has_context})

        avg_retrieval = sum(r["top_score"] for r in query_results) / len(query_results)
        coverage = sum(1 for r in query_results if r["has_context"]) / len(query_results)

        entry = {
            "heading_bias": mode,
            "num_chunks": len(all_chunks),
            "avg_retrieval": round(avg_retrieval, 4),
            "coverage": round(coverage, 3),
        }
        results.append(entry)
        print(f"    avg_score={avg_retrieval:.4f}  coverage={coverage:.1%}")

    out_path = "data/ablation_heading_bias.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    best = max(results, key=lambda r: r["avg_retrieval"])
    print(f"\nBest config: heading_bias={best['heading_bias']}, avg_score={best['avg_retrieval']:.4f}")

    return results


# Study 6
def study_temperature():
    """Test different LLM temperature values."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 6: Temperature")
    print("=" * 60)

    import rag.embedder as embedder
    if embedder._index is None:
        load_faiss_index()

    temperatures = [0.1, 0.3, 0.5, 0.7]

    results = []

    for temp in temperatures:
        print(f"\n  Testing temperature={temp}...")

        query_results = []
        for bq in CORE_QUERIES:
            query = bq["query"]
            ref = bq.get("reference_answer", "")

            expanded = _expand_query(query)
            query_vec = embed_texts([expanded])
            scores, indices = embedder._index.search(query_vec, 5)

            chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < 0.35:
                    continue
                chunk = embedder._chunks[idx]
                chunks.append({
                    "text": chunk["text"],
                    "source_url": chunk["source_url"],
                    "page_title": chunk["page_title"],
                    "score": float(score),
                })

            from huggingface_hub import InferenceClient
            from rag.generator import SYSTEM_PROMPT
            client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

            context_block = "\n\n---\n\n".join(
                f"[Source: {c['page_title']}]\n{c['text']}" for c in chunks
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ]

            try:
                response = client.chat_completion(
                    messages=messages,
                    model="HuggingFaceH4/zephyr-7b-beta",
                    max_tokens=1024,
                    temperature=temp,
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print(f"    Error: {e}")
                answer = ""

            similarity = _compare_answers(answer, ref)
            top_score = chunks[0]["score"] if chunks else 0.0

            query_results.append({
                "top_score": top_score,
                "recall": similarity["recall"],
            })

            time.sleep(1)

        avg_retrieval = sum(r["top_score"] for r in query_results) / len(query_results)
        avg_recall = sum(r["recall"] for r in query_results) / len(query_results)

        entry = {
            "temperature": temp,
            "avg_retrieval": round(avg_retrieval, 4),
            "avg_recall": round(avg_recall, 3),
        }
        results.append(entry)
        print(f"    avg_score={avg_retrieval:.4f}  avg_recall={avg_recall:.3f}")

    out_path = "data/ablation_temperature.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    best = max(results, key=lambda r: r["avg_recall"])
    print(f"\nBest config: temperature={best['temperature']}, recall={best['avg_recall']:.3f}")

    return results


# Main 
def main():
    parser = argparse.ArgumentParser(description="KCL RAG Chatbot — Ablation Studies")
    parser.add_argument("--study", type=str, default="all",
                        choices=["chunk_geometry", "info_volume", "query_expansion",
                                 "similarity_threshold", "heading_bias", "temperature", "all"],
                        help="Which ablation study to run")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.study in ("chunk_geometry", "all"):
        study_chunk_geometry()

    if args.study in ("heading_bias", "all"):
        study_heading_bias()

    if args.study in ("query_expansion", "all"):
        study_query_expansion()

    if args.study in ("similarity_threshold", "all"):
        study_similarity_threshold()

    if args.study in ("info_volume", "all"):
        study_info_volume()

    if args.study in ("temperature", "all"):
        study_temperature()

    print("\n" + "=" * 60)
    print("All requested ablation studies complete!")
    print(f"Results saved in {DATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
