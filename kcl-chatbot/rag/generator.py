import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import LLM_PROVIDER

SYSTEM_PROMPT = """You are the KCL Student Services Assistant, a helpful chatbot for King's College London students.

RULES:
1. Answer the student's question using ONLY the provided context passages below.
2. If the context does not contain enough information, say so honestly and suggest contacting Student Services directly.
3. Be THOROUGH — include all relevant details, steps, and specific instructions from the context.
4. If the context contains step-by-step instructions, include ALL steps in your answer.
5. Include specific names of services, systems, and forms mentioned in the context.
6. Where relevant, mention the source page so the student can find more detail.
7. Do NOT make up information not present in the context.
8. Do NOT answer questions unrelated to KCL Student Services.
9. Keep your tone friendly and clear.
10. If the context is limited, give a short answer based only on what is available. Do NOT pad your response with general advice or information not found in the context.
11. Do NOT respond with questions. Do NOT rephrase the student's question. Start your response with the answer directly."""

def _generate_local(query, chunks):
    """
    Extractive fallback — no API key needed.
    Returns the best matching chunks with source attribution.
    Good for testing the retrieval pipeline.
    """
    if not chunks:
        return (
            "I'm sorry, I couldn't find any relevant information for your query "
            "in the KCL Student Services knowledge base. Please try rephrasing "
            "your question or contact Student Services directly."
        )

    parts = ["Based on the KCL Student Services information I found:\n"]
    seen_sources = set()
    for c in chunks[:3]:
        text = c["text"]
        if len(text) > 600:
            text = text[:600].rsplit(" ", 1)[0] + " ..."
        source = c.get("page_title", c.get("source_url", ""))
        parts.append(f"• {text}")
        if source and source not in seen_sources:
            parts.append(f"  (Source: {source})")
            seen_sources.add(source)
        parts.append("")

    parts.append(
        "For more details, visit https://self-service.kcl.ac.uk/ "
        "or contact Student Services directly."
    )
    return "\n".join(parts)


def _generate_huggingface(query, chunks):
    """Call HuggingFace Inference API directly."""
    from huggingface_hub import InferenceClient
    import os

    client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['page_title']}]\n{c['text']}" for c in chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
    ]

    response = client.chat_completion(
        messages=messages,
        model="HuggingFaceH4/zephyr-7b-beta",
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content

#Tested two-step summarisation, but did not work as evaluation showed the average recall to be much less.
#Likely due to the free LLM.
def _summarise_response(answer):
    """Clean up a long answer by removing repetition while keeping all key details."""
    from huggingface_hub import InferenceClient
    import os

    client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    messages = [
        {"role": "system", "content": "Rewrite the following response to be clearer and more organised. Remove any repetition or redundant information, but keep ALL important details, steps, names, and instructions. Do not remove any factual content. The response should be shorter than the original but still comprehensive."},
        {"role": "user", "content": answer},
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model="HuggingFaceH4/zephyr-7b-beta",
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return answer

def _clean_response(answer):
    """Remove LLM artifacts, formatting junk, and repetition loops."""
    import re
    
    # Remove common LLM artifacts
    answer = answer.replace("[/INST]", "").replace("[/ASS]", "").replace("[/STU]", "").replace("[/USER]", "")
    
    # Remove [Source: ...] references
    answer = re.sub(r"\[Source:.*?\]", "", answer)
    
    # Remove lines starting with "Passage:" or "Context:" or "Answer:"
    lines = answer.split("\n")
    lines = [l for l in lines if not l.strip().startswith(("Passage:", "Context:", "Answer:"))]
    answer = "\n".join(lines)
    
    # Remove --- separators
    answer = answer.replace("---", "")
    
    # Remove unicode zero-width characters
    answer = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', answer)
    
    # Remove repetition loops
    sentences = answer.replace("?", "?.").replace("!", "!.").split(".")
    seen = set()
    clean = []
    for s in sentences:
        s_stripped = s.strip().lower()
        if not s_stripped or len(s_stripped) < 10:
            continue
        is_duplicate = False
        for prev in seen:
            s_words = set(s_stripped.split())
            p_words = set(prev.split())
            if not s_words:
                continue
            overlap = len(s_words & p_words) / len(s_words)
            if overlap > 0.7:
                is_duplicate = True
                break
        if not is_duplicate:
            clean.append(s.strip())
            seen.add(s_stripped)
    
    # Remove leading questions
    first_answer = 0
    for i, s in enumerate(clean):
        if not s.endswith("?"):
            first_answer = i
            break
    clean = clean[first_answer:]
    
    result = ". ".join(clean)
    if result and not result.endswith(".") and not result.endswith("?"):
        result += "."
    
    # Clean up whitespace
    result = re.sub(r"\s+", " ", result).strip()
    
    if len(result) < 50:
        return "I found some relevant information but couldn't generate a clear answer. Please visit https://self-service.kcl.ac.uk/ or contact Student Services directly."
    
    return result

def generate_response(query, chunks):
    """
    Generate a response using the configured LLM provider.
    Returns a dict with answer, sources, scores, and provider name.
    """
    provider = LLM_PROVIDER

    if provider == "huggingface":
        if not chunks:
            answer = ("I'm sorry, I couldn't find any relevant information for your query "
                      "in the KCL Student Services knowledge base. Please try rephrasing "
                      "your question or contact Student Services directly.")
        else:
            answer = _generate_huggingface(query, chunks)
            answer = _clean_response(answer)
    else:
        answer = _generate_local(query, chunks)

    # Deduplicate sources
    seen = set()
    sources = []
    for c in chunks:
        url = c.get("source_url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({"url": url, "title": c.get("page_title", "")})

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_scores": [c["score"] for c in chunks],
        "provider": provider,
    }