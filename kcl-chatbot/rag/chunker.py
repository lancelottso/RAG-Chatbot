import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHUNK_SIZE, CHUNK_OVERLAP, SCRAPED_PAGES_PATH, CHUNKS_PATH


def _split_text(text, max_tokens, overlap):
    """Split text into word-based chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += max_tokens - overlap
    return chunks


def chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Convert scraped pages into a flat list of chunk dicts.
    Each chunk keeps its source URL and page title for attribution.
    """
    all_chunks = []
    chunk_id = 0

    for page in pages:
        body = page.get("body", "")
        if not body or len(body.split()) < 20:
            continue

        # Build heading context to prepend to first chunk of each page
        heading_context = ""
        if page.get("title"):
            heading_context += page["title"] + ". "
        if page.get("headings"):
            heading_context += " — ".join(page["headings"][:5]) + ". "

        raw_chunks = _split_text(body, chunk_size, chunk_overlap)

        for text in raw_chunks:
            # Prepend heading context to the first chunk of each page
            enriched = heading_context + text if text == raw_chunks[0] else text
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": enriched,
                "source_url": page.get("url", ""),
                "page_title": page.get("title", ""),
                "headings": page.get("headings", []),
            })
            chunk_id += 1

    print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


def save_chunks(chunks, path=CHUNKS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks to {path}")


def load_chunks(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    with open(SCRAPED_PAGES_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)
    chunks = chunk_pages(pages)
    save_chunks(chunks)