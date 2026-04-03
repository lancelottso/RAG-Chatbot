import json
import os
import sys
import time
from datetime import datetime
from rag.generator import generate_response

sys.path.insert(0, os.path.dirname(__file__))
from rag import answer_query

BENCHMARK_QUERIES = [
    {
        "query": "How do I apply for extenuating circumstances?",
        "expected_topic": "extenuating circumstances",
        "reference_answer": "Log on to Student Records. Select 'Student Home Page' within the banner at the top of the homepage. Select 'My Mitigating Circumstances' within the 'My Modules' area. On the Mitigating Circumstances homepage, you will find links to user guides, videos and other Student Services Online articles. Before submitting a new request, please read this guidance and familiarise yourself with the Mitigating Circumstances Policy. To begin completing a request, select 'New Request' within the 'Submit a new Mitigating Circumstances request' area.",
    },
    {
        "query": "What happens if I miss an exam?",
        "expected_topic": "missed exam",
        "reference_answer": "If you miss an exam due to unforeseen circumstances, you should contact your department or faculty as soon as possible to understand the specific rules regarding missed assessments for your programme.",
    },
    {
        "query": "How do I change my programme or module?",
        "expected_topic": "programme change",
        "reference_answer": "If you change your mind about the modules that you wish to take, then you should contact your department or faculty initially to discuss your options with them. Not all departments are able to permit changes to modules and many departments will have a deadline by which they can consider such changes.",
    },
    {
        "query": "Where can I find wellbeing support?",
        "expected_topic": "wellbeing",
        "reference_answer": "King's offers an online support service called Spectrum.Life that you can use at any time. The service is free and confidential, available 24 hours a day, 7 days a week. To join, visit Spectrum.Life to register, enter your name and King's email address, choose a password, and make sure the Organisation code is set to KCL.",
    },
    {
        "query": "How do I get a replacement student ID card?",
        "expected_topic": "student ID",
        "reference_answer": "If you lose your card, report it immediately to Student Services or Security so it can be deactivated. Visit one of the service desks where they can check your student record and produce a replacement ID card while you wait. Bring photographic ID such as your passport. If replacing a damaged card, you will be given a new one in exchange for the damaged card.",
    },
    {
        "query": "What is the deadline for tuition fee payment?",
        "expected_topic": "tuition fees",
        "reference_answer": "Details on when you can pay your tuition fees can be found on the Credit Control webpages.",
    },
    {
        "query": "How do I request a transcript?",
        "expected_topic": "transcript",
        "reference_answer": "The transcript is issued only to students who have completed their course and been awarded. Once you have been awarded, you are considered a former and graduated student and alumni of King's. If you are still studying or completing assessments, refer to the information for current students.",
    },
    {
        "query": "Can I get an extension for my coursework?",
        "expected_topic": "extension",
        "reference_answer": "You must submit a Mitigating Circumstances Form. Extensions are not automatically granted and you must submit a separate form for each assessment. Log in to Student Records, select Student Home Page, select My Mitigating Circumstances, select New Request, choose Extension Granted by PAA from the Nature of Circumstances dropdown, and complete the form.",
    },
    {
        "query": "How do I enrol for the new academic year?",
        "expected_topic": "enrolment",
        "reference_answer": "You will receive an email inviting you to complete re-enrolment. Log into Student Records and navigate to the My Enrolment page. Complete the online enrolment task as soon as you can, as access to library services, KEATS, and funding will be affected if you re-enrol late. Then complete essential training by navigating to the My Training tab in Student Records.",
    },
    {
        "query": "What is the capital of France?",
        "expected_topic": "__out_of_domain__",
        "reference_answer": "",
    },
]

def _compare_answers(generated, reference):
    """
    Compare generated answer against reference using word overlap (Jaccard similarity).
    Returns a score between 0 and 1.
    """
    if not reference:
        return None

    # Tokenise into word sets
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())

    # Remove common stop words that don't carry meaning
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

    gen_words -= stop_words
    ref_words -= stop_words

    if not ref_words:
        return None

    # Jaccard similarity: overlap / union
    overlap = gen_words & ref_words
    union = gen_words | ref_words
    jaccard = len(overlap) / len(union) if union else 0.0

    # Also calculate recall: how much of the reference is covered
    recall = len(overlap) / len(ref_words) if ref_words else 0.0

    return {"jaccard": round(jaccard, 3), "recall": round(recall, 3)}

def _check_faithfulness(answer, chunks):
    """
    Check what proportion of the answer's content words appear in the retrieved chunks.
    A high score means the answer is grounded in the context (faithful).
    A low score suggests the LLM may be hallucinating.
    """
    if not chunks or not answer.strip():
        return {"faithfulness": 0.0}

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
                  "other", "some", "such", "no", "only", "own", "same", "about",
                  "also", "just", "please", "here", "there", "very", "well",
                  "based", "following", "help", "need", "want", "like", "make",
                  "know", "see", "get", "go", "come", "take", "give", "find",
                  "think", "say", "tell", "ask", "use", "try", "keep", "let",
                  "begin", "seem", "show", "hear", "play", "run", "move", "live"}

    # Combine all chunk text into one set of words
    context_words = set()
    for c in chunks:
        context_words.update(c["text"].lower().split())
    context_words -= stop_words

    # Get answer words
    answer_words = set(answer.lower().split()) - stop_words

    if not answer_words:
        return {"faithfulness": 0.0}

    # What proportion of answer words appear in the context?
    grounded = answer_words & context_words
    faithfulness = len(grounded) / len(answer_words)

    return {"faithfulness": round(faithfulness, 3)}

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
        from rag.embedder import search
        chunks = search(query)
        result = generate_response(query, chunks)
        elapsed = time.time() - start

        scores = result["retrieval_scores"]
        top_score = scores[0] if scores else 0.0
        has_context = len(scores) > 0

        # Compare against reference answer
        ref = bq.get("reference_answer", "")
        similarity = _compare_answers(result["answer"], ref)
        faithfulness = _check_faithfulness(result["answer"], chunks)

        entry = {
            "query": query,
            "expected_topic": expected,
            "top_score": round(top_score, 4),
            "num_sources": len(result["sources"]),
            "has_context": has_context,
            "answer_length": len(result["answer"]),
            "latency_s": round(elapsed, 2),
            "similarity": similarity,
            "faithfulness": faithfulness["faithfulness"],
        }
        
        results.append(entry)
        sim_str = f"jaccard={similarity['jaccard']:.2f}  recall={similarity['recall']:.2f}" if similarity else "n/a"
        print(f"  top={top_score:.3f}  {sim_str}  faithful={faithfulness['faithfulness']:.2f}  latency={elapsed:.1f}s\n")

    # Summary
    in_domain = [r for r in results if r["expected_topic"] != "__out_of_domain__"]
    avg_top = sum(r["top_score"] for r in in_domain) / len(in_domain) if in_domain else 0
    coverage = sum(1 for r in in_domain if r["has_context"]) / len(in_domain) * 100 if in_domain else 0

    # Calculate average similarity scores
    sim_results = [r["similarity"] for r in in_domain if r["similarity"]]
    avg_jaccard = sum(s["jaccard"] for s in sim_results) / len(sim_results) if sim_results else 0
    avg_recall = sum(s["recall"] for s in sim_results) / len(sim_results) if sim_results else 0
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0
    avg_faithfulness = sum(r["faithfulness"] for r in in_domain) / len(in_domain) if in_domain else 0

    print(f"\n{'─'*60}")
    print("Summary")
    print(f"  In-domain queries  : {len(in_domain)}")
    print(f"  Avg top-1 score    : {avg_top:.3f}")
    print(f"  Coverage (%)       : {coverage:.1f}%")
    print(f"  Avg Jaccard sim    : {avg_jaccard:.3f}")
    print(f"  Avg Recall         : {avg_recall:.3f}")
    print(f"  Avg latency        : {avg_latency:.2f}s")
    print(f"  Avg Faithfulness   : {avg_faithfulness:.3f}")
    print(f"{'─'*60}\n")

    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Results saved to data/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()