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
        "query": "How can I make an academic appeal?",
        "expected_topic": "academic appeal",
        "reference_answer": "The appeals process involves two stages. Stage 1 concerns decisions of the Faculty Assessment Board. Stage 2 is made to Student Conduct and Appeals if not satisfied with Stage 1. From 8 July 2025, Stage 1 appeals are submitted through Student Records. Log in and navigate to My Academic Appeals in My Modules, select New Request, provide supporting information, and submit within 15 working days of ratified results.",
    },
    {
        "query": "What are mitigating circumstances?",
        "expected_topic": "mitigating circumstances",
        "reference_answer": "Mitigating circumstances are defined as significant and unavoidable events beyond a student's control which may affect performance in a summative assessment. The Mitigating Circumstances Policy applies to students on taught programmes. Marks will never be raised due to mitigating circumstances.",
    },
    {
        "query": "What evidence do I need for my mitigating circumstances application?",
        "expected_topic": "mitigating evidence",
        "reference_answer": "Evidence must be in writing, written by an independent third party such as your personal doctor, supportive of your claim, signed and dated, relevant to the affected period, and written in English or accompanied by an English translation.",
    },
    {
        "query": "How do I apply for 5-working-day extensions for coursework?",
        "expected_topic": "extensions",
        "reference_answer": "Submit a Mitigating Circumstances Form. Extensions are not automatically granted and you must submit a separate form for each assessment. Log in to Student Records, select Student Home Page, select My Mitigating Circumstances, select New Request, choose Extension Granted by PAA from the dropdown, and complete the form.",
    },
    {
        "query": "What items can I take into my exams?",
        "expected_topic": "exam items",
        "reference_answer": "Allowed items include student ID card, pens in blue or black ink, pencils for multiple choice only, erasers, rulers, sharpeners, calculator if permitted, sealable clear plastic wallet no larger than A4, non-carbonated non-alcoholic drinks not in glass bottles, clear bottles with no labels. Only Casio fx-83 or fx-85 calculators are approved.",
    },
    {
        "query": "How can I withdraw from my programme?",
        "expected_topic": "withdrawal",
        "reference_answer": "Read about and follow the process for registration changes on the relevant intranet section for undergraduate, postgraduate taught, or postgraduate research. Then complete the Change of Circumstances form on Student Records selecting Initiate Change of Circumstances.",
    },
    {
        "query": "What are Personalised Assessment Arrangements?",
        "expected_topic": "PAA",
        "reference_answer": "Personalised Assessment Arrangements provide an environment that gives all students an equal opportunity for assessment. If you have a diagnosed Specific Learning Difference, neurodiverse disability, long term medical or mental health condition or other disability affecting written assessments, you can apply by registering with the Disability and Inclusion team.",
    },
    {
        "query": "How do I upload my photo for my student ID card?",
        "expected_topic": "ID photo",
        "reference_answer": "Log into Student Records with your K number and password. Go to My online Tasks on the left side. Select the message with subject 2025-26 Photo upload. Upload your photo. Image must be in JPEG or PNG format, between 40KB and 1MB, at least 320 by 400 pixels, a recent clear photo with no other objects or people.",
    },
    {
        "query": "How can I take a break from my course?",
        "expected_topic": "interruption",
        "reference_answer": "Contact your personal tutor or supervisor to discuss your situation. Read about practical implications, fees and funding impact, and visa impact. Speak to Student Support and Wellbeing services. Process your request via the online form in your student record by selecting Initiate Change of Circumstances.",
    },
    {
        "query": "How do I access my timetable?",
        "expected_topic": "timetable",
        "reference_answer": "You can view your timetable once you have completed enrolment by logging into My Timetable using your K number and password.",
    },
    {
        "query": "How does the university handle academic misconduct?",
        "expected_topic": "academic misconduct",
        "reference_answer": "The university takes examination and assessment offences very seriously and sets out procedures in the King's Academic Manual, the Academic Misconduct Policy and the Academic Misconduct Procedures.",
    },
    {
        "query": "How can I make a complaint to the university?",
        "expected_topic": "complaint",
        "reference_answer": "Complaints should first be raised informally with a relevant individual such as a Personal Tutor, Senior Tutor, or Head of Department for a local level investigation at Stage 1 of the Student Complaints Procedure. If the complaint relates to another student's conduct, it will be considered under the Misconduct Regulations.",
    },
    {
        "query": "How do I register for my modules?",
        "expected_topic": "module registration",
        "reference_answer": "Follow the links in the email from Student Admin Services to log into Student Records. Follow the User Help provided by your department. Modules available will be listed in collections with instructions on how many to select. Selected modules appear in a shopping basket on the right side of the page.",
    },
    {
        "query": "How can King's Careers and Employability help me?",
        "expected_topic": "careers",
        "reference_answer": "At the start of every academic year, King's asks short questions during enrolment about your career planning stage, plans after study, desired employment sector, and work experience. These help the Careers and Employability Service give you support and help you understand where you are on your career journey.",
    },
    {
        "query": "How do I update my contact details on my student record?",
        "expected_topic": "contact details",
        "reference_answer": "Log in to Student Records, go to Student Home Page, navigate to My Personal Details, click View and Edit My Contact Information. You can edit your home address and term-time address. Choose the address to update, click Edit, amend it, and click Save. For email, click Edit next to Home address and update your personal email.",
    },
    {
        "query": "When will my exam results be available?",
        "expected_topic": "exam results",
        "reference_answer": "Results are released on Student Records after the assessment board meets. Period 1 January assessments in March to April, Period 2 May assessments in July, Period 3 August assessments in September. Most postgraduate taught assessment boards meet between October and December after the course end date.",
    },
    {
        "query": "How do I change my name on my student record?",
        "expected_topic": "name change",
        "reference_answer": "You do not need formal documents to change your Known as name. To change your formal name, contact the Student Records team via the enquiry form and attach official documents showing your preferred name.",
    },
    {
        "query": "What is the academic calendar?",
        "expected_topic": "academic calendar",
        "reference_answer": "The academic calendar captures dates for semesters and other key dates across the academic year which runs from September to August. These dates apply to all faculties and the majority of undergraduate and postgraduate taught programmes.",
    },
    {
        "query": "When is the graduation ceremony this year?",
        "expected_topic": "graduation",
        "reference_answer": "July 2026 ceremonies are held at Southwark Cathedral and Royal Festival Hall. Faculty of Dentistry on Tuesday 7 July, Faculty of Life Sciences and Medicine on Wednesday 8 July at Southwark Cathedral. At Royal Festival Hall, ceremonies run from Monday 27 July to Friday 31 July for remaining faculties including Law, Arts and Humanities, Natural Mathematical and Engineering Sciences, Social Science, Nursing, Psychology, and King's Business School.",
    },
    {
        "query": "What is the Right to Study and why do I need to do it?",
        "expected_topic": "right to study",
        "reference_answer": "UKVI policy states universities must determine a student's right to study before full enrolment. If you do not hold a British or Irish passport, you must provide evidence of current immigration permission to be in the UK. It takes 5 to 10 minutes to complete online and up to 3 working days for review.",
    },
    {
        "query": "What is the capital of France?",
        "expected_topic": "__out_of_domain__",
        "reference_answer": "",
    }
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