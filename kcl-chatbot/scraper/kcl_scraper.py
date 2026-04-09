import json
import re
import time
import os
import sys
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment, NavigableString
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import KCL_BASE_URL, MAX_PAGES, SCRAPE_DELAY, SCRAPED_PAGES_PATH, DATA_DIR


def _is_valid_url(url, base_domain):
    """Check if URL belongs to target domain and is an HTML page."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if base_domain not in parsed.netloc:
        return False
    skip_ext = (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                ".css", ".js", ".zip", ".xlsx", ".docx", ".pptx")
    if any(parsed.path.lower().endswith(ext) for ext in skip_ext):
        return False
    # Skip category index pages - they contain only lists of question links, no actual content
    if "/category/" in parsed.path:
        return False
    return True


def _clean_text(raw):
    """Collapse whitespace."""
    return re.sub(r"\s+", " ", raw).strip()


def _extract_content(soup):
    """
    Extract meaningful content from a page, removing nav/footer/sidebar chrome.
    Returns dict with title, headings, and body text.
    """
    # Remove unwanted elements
    for tag in soup.find_all(["nav", "footer", "header", "script",
                              "style", "noscript", "iframe"]):
        tag.decompose()

    # Remove elements with common chrome class/id names
    chrome_pattern = re.compile(
        r"(nav|footer|header|sidebar|menu|cookie|banner|breadcrumb|"
        r"skip-link|search-bar|social|share|pagination)", re.IGNORECASE
    )
    for tag in soup.find_all(attrs={"class": chrome_pattern}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": chrome_pattern}):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Get title
    title_tag = soup.find("title")
    title = _clean_text(title_tag.get_text()) if title_tag else ""

    # Find main content area
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("div", class_=re.compile(r"(content|main|body)", re.I))
        or soup.body
    )

    if main is None:
        return {"title": title, "headings": [], "body": ""}

    # Handle collapsible Q&A panels by formatting them clearly
    # This prevents the LLM from seeing a jumble of questions mixed with answers
    for panel in main.find_all("section", class_="collapsible_panel"):
        title_div = panel.find("div", class_="collapsible_panel_title")
        body_div = panel.find("div", class_="collapsible_panel_body")
        if title_div and body_div:
            title_text = _clean_text(title_div.get_text())
            body_text = _clean_text(body_div.get_text())
            replacement = NavigableString(f" Question: {title_text} Answer: {body_text} ")
            panel.replace_with(replacement)

    headings = [_clean_text(h.get_text()) for h in main.find_all(re.compile(r"^h[1-6]$"))]
    body = _clean_text(main.get_text())

    # Remove common KCL page junk
    junk_patterns = [
        r"Keywords:.*?(?=\n|$)",
        r"Related Articles\s*\(\d+\).*?(?=\n|$)",
        r"Comments\s*\(\d+\).*?(?=\n|$)",
        r"Can't find your answer\?.*?(?=\n|$)",
        r"Was this article helpful\?.*?(?=\n|$)",
        r"Provide us some feedback here.*?(?=\n|$)",
        r"Get in touch via our contact form.*?(?=\n|$)",
        r"Email:.*?(?=\n|$)",
    ]
    for pattern in junk_patterns:
        body = re.sub(pattern, "", body, flags=re.IGNORECASE)
    body = _clean_text(body)

    return {"title": title, "headings": headings, "body": body}


def scrape_kcl(start_url=KCL_BASE_URL, max_pages=MAX_PAGES, delay=SCRAPE_DELAY):
    """
    BFS crawl starting at start_url. Returns list of page dicts:
    [{"url": ..., "title": ..., "headings": [...], "body": ...}, ...]
    """
    base_domain = urlparse(start_url).netloc
    visited = set()
    queue = deque([start_url])
    pages = []

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; KCL-RAG-Chatbot/1.0; student project)"
    })

    pbar = tqdm(total=max_pages, desc="Scraping pages")

    while queue and len(pages) < max_pages:
        url = queue.popleft().split("#")[0]  # strip fragment
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except requests.RequestException as exc:
            tqdm.write(f"  [WARN] Failed: {url}: {exc}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        content = _extract_content(soup)

        if len(content["body"]) > 100:
            content["url"] = url
            pages.append(content)
            pbar.update(1)

        # Discover new links
        for a_tag in soup.find_all("a", href=True):
            href = urljoin(url, a_tag["href"]).split("#")[0]
            if href not in visited and _is_valid_url(href, base_domain):
                queue.append(href)

        time.sleep(delay)

    pbar.close()
    print(f"\nScraped {len(pages)} pages from {start_url}")
    return pages


def save_pages(pages, path=SCRAPED_PAGES_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    print(f"Saved scraped pages to {path}")


if __name__ == "__main__":
    pages = scrape_kcl()
    save_pages(pages)