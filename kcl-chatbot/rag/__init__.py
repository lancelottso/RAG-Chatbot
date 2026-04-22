from .embedder import search
from .generator import generate_response
from config import TOP_K, SIMILARITY_THRESHOLD


def answer_query(query, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    """End-to-end: retrieve relevant chunks then generate a response."""
    chunks = search(query, top_k=top_k, threshold=threshold)
    return generate_response(query, chunks)