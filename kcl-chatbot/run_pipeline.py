import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import SCRAPED_PAGES_PATH, CHUNKS_PATH, FAISS_INDEX_PATH, DATA_DIR
from scraper.kcl_scraper import scrape_kcl, save_pages
from rag.chunker import chunk_pages, save_chunks
from rag.embedder import build_faiss_index


def main():
    parser = argparse.ArgumentParser(description="KCL RAG Chatbot — Data Pipeline")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping; use existing scraped_pages.json")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Scrape
    if args.skip_scrape:
        print("\n[1/3] Loading existing scraped pages...")
        with open(SCRAPED_PAGES_PATH, "r", encoding="utf-8") as f:
            pages = json.load(f)
        print(f"  Loaded {len(pages)} pages.")
    else:
        print("\n[1/3] Scraping KCL Student Services...")
        pages = scrape_kcl()
        save_pages(pages)

    # Step 2: Chunk
    print("\n[2/3] Chunking pages...")
    chunks = chunk_pages(pages)
    save_chunks(chunks)

    # Step 3: Embed & Index
    print("\n[3/3] Embedding chunks & building FAISS index...")
    build_faiss_index(chunks)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"  Pages scraped : {len(pages)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  FAISS index   : {FAISS_INDEX_PATH}.index")
    print(f"\nStart the chatbot:  python3 chatbot.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()