# KCL Student Services RAG Chatbot

A chatbot that answers student queries about KCL Student Services. It scrapes content from the KCL self-help website, stores it in a vector database, and uses an LLM to generate answers.

Built as part of my BSc Computer Science Individual Project at King's College London.

## How to run

### 1. Set up
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create a `.env` file
```
FLASK_SECRET_KEY=change-this-to-a-random-secret-key
FLASK_DEBUG=True
KCL_BASE_URL=https://self-service.kcl.ac.uk/
MAX_PAGES=200
SCRAPE_DELAY=1.0
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss_index
CHUNKS_PATH=data/chunks.json
CHUNK_SIZE=300
CHUNK_OVERLAP=75
TOP_K=5
SIMILARITY_THRESHOLD=0.35
LLM_PROVIDER=huggingface
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

You can get a free HuggingFace token at https://huggingface.co/settings/tokens

If you don't have a token, set `LLM_PROVIDER=local` to use the extractive fallback mode (no API needed).

### 3. Build the knowledge base
```bash
python3 run_pipeline.py
```

This scrapes the KCL website, chunks the text, and builds the FAISS index. Takes a few minutes.

### 4. Run the chatbot
```bash
python3 chatbot.py
```

Then go to http://localhost:5000

### 5. Run evaluation
```bash
python3 evaluate.py
```

### 6. Run ablation studies
```bash
python3 ablation.py --study all
```

Or run one at a time: `chunk_geometry`, `info_volume`, `query_expansion`, `similarity_threshold`, `heading_bias`, `temperature`.

## Project structure

- `chatbot.py` — Flask web app
- `config.py` — loads settings from .env
- `run_pipeline.py` — scrapes site, chunks text, builds FAISS index
- `evaluate.py` — runs benchmark queries and measures accuracy
- `ablation.py` — tests different configurations
- `scraper/` — web scraper for KCL Student Services
- `rag/` — chunking, embedding, retrieval, and response generation
- `static/` and `templates/` — chat interface frontend
- `data/` — generated files (scraped pages, chunks, FAISS index, evaluation & ablation results)

## Tech stack

- Python, Flask
- BeautifulSoup (web scraping)
- sentence-transformers / all-MiniLM-L6-v2 (embeddings)
- FAISS (vector search)
- HuggingFace Inference API / Zephyr-7B-beta (LLM)