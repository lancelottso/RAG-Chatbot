import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, CHUNKS_PATH

# Global state for lazy loading
_model = None
_index = None
_chunks = None


def _get_model():
    """Lazy-load the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts, batch_size=64):
    """Return L2-normalised embeddings for a list of texts."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalised → dot product = cosine similarity
    )
    return embeddings.astype("float32")


def build_faiss_index(chunks, save_path=FAISS_INDEX_PATH):
    """
    Embed all chunks and build a FAISS inner-product index.
    Saves the index file and a metadata JSON file.
    """
    import faiss

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalised vectors = cosine
    index.add(embeddings)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    faiss.write_index(index, save_path + ".index")

    # Save metadata mapping (chunk_id, source, title for each row)
    meta = [{"chunk_id": c["chunk_id"], "source_url": c["source_url"],
             "page_title": c["page_title"]} for c in chunks]
    with open(save_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"FAISS index saved: {save_path}.index ({index.ntotal} vectors, dim={dim})")


def load_faiss_index(index_path=FAISS_INDEX_PATH):
    """Load the FAISS index and chunk data into memory."""
    import faiss
    global _index, _chunks

    _index = faiss.read_index(index_path + ".index")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        _chunks = json.load(f)

    print(f"Loaded FAISS index ({_index.ntotal} vectors) and {len(_chunks)} chunks")
    return _index, _chunks


def search(query, top_k=5, threshold=0.35):
    """
    Embed a query and return the top-k most similar chunks.
    Filters out results below the similarity threshold.
    """
    global _index, _chunks
    if _index is None or _chunks is None:
        load_faiss_index()

    query_vec = embed_texts([query])
    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        chunk = _chunks[idx]
        results.append({
            "text": chunk["text"],
            "source_url": chunk["source_url"],
            "page_title": chunk["page_title"],
            "score": float(score),
        })
    return results


if __name__ == "__main__":
    from rag.chunker import load_chunks
    chunks = load_chunks()
    build_faiss_index(chunks)