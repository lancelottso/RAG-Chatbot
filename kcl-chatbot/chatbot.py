import os
import sys
import json
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify

from config import FLASK_SECRET_KEY, FLASK_DEBUG, FAISS_INDEX_PATH, CHUNKS_PATH
from rag import answer_query

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUERY_LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "query_log.jsonl")


def _log_query(query, result):
    """Log anonymised queries for evaluation (FR10)."""
    os.makedirs(os.path.dirname(QUERY_LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer_preview": result["answer"][:200],
        "num_sources": len(result["sources"]),
        "top_score": result["retrieval_scores"][0] if result["retrieval_scores"] else None,
        "provider": result["provider"],
    }
    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _index_exists():
    return os.path.exists(FAISS_INDEX_PATH + ".index") and os.path.exists(CHUNKS_PATH)


@app.route("/")
def index():
    return render_template("chat.html", index_ready=_index_exists())


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter a question."}), 400

    if not _index_exists():
        return jsonify({"error": "Knowledge base not built yet. Run: python run_pipeline.py"}), 503

    try:
        result = answer_query(query)
        _log_query(query, result)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Error processing query")
        return jsonify({"error": f"Something went wrong: {exc}"}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "index_ready": _index_exists()})


if __name__ == "__main__":
    if not _index_exists():
        logger.warning("FAISS index not found. Run: python run_pipeline.py")
    app.run(host="0.0.0.0", port=5000, debug=FLASK_DEBUG)