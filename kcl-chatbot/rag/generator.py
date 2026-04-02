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
9. Keep your tone friendly and clear."""


def _build_prompt(query, context_chunks):
    """Assemble the full prompt with system instructions and retrieved context."""
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['page_title']}]\n{c['text']}" for c in context_chunks
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== STUDENT QUESTION ===\n{query}\n\n"
        f"=== YOUR ANSWER ==="
    )


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

def _generate_openai(query, chunks):
    """Call OpenAI (needs OPENAI_API_KEY in .env)."""
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=512)
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['page_title']}]\n{c['text']}" for c in chunks
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context_block}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


def generate_response(query, chunks):
    """
    Generate a response using the configured LLM provider.
    Returns a dict with answer, sources, scores, and provider name.
    """
    provider = LLM_PROVIDER

    if provider == "openai":
        answer = _generate_openai(query, chunks)
    elif provider == "huggingface":
        answer = _generate_huggingface(query, chunks)
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