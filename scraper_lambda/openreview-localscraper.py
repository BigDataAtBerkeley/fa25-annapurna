"""
openreview_llm_scraper.py

Every run:
- Loads the OpenReview search for "LLM" (papers only)
- Scrolls to the bottom to load all results
- Visits each forum page, extracts metadata + PDF link
- Uploads PDF to S3 if not already present
- Sends a JSON task to SQS (FIFO-ready)
- Repeats every 24 hours (configurable via SLEEP_HOURS)

"""

import os
import time
import json
import uuid
import boto3
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ---------- config ----------
SEARCH_TERM = os.getenv("SEARCH_TERM", "LLM")
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "20"))       # cap to keep tests quick
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"           # skip actual downloads
USE_MANAGER = os.getenv("USE_MANAGER", "0") == "1"   # use webdriver-manager if needed

OPENREVIEW_SEARCH = (
    f"https://openreview.net/search?content=all&group=all&source=forum&term={SEARCH_TERM}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

DOWNLOAD_DIR = pathlib.Path("./downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ---------- helpers ----------
def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1200,2000")
    if USE_MANAGER:
        from webdriver_manager.chrome import ChromeDriverManager
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
    else:
        driver = webdriver.Chrome(options=opts)
    return driver

def parse_forum_id(url: str):
    try:
        q = parse_qs(urlparse(url).query)
        return (q.get("id") or q.get("forum") or [None])[0]
    except Exception:
        return None

def get_forum_urls(driver: webdriver.Chrome, cap: int | None = None):
    driver.get(OPENREVIEW_SEARCH)

    # scroll to load all results (lazy/infinite load)
    last_h, stable = 0, 0
    while stable < 3:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.0)
        h = driver.execute_script("return document.body.scrollHeight")
        if h == last_h:
            stable += 1
        else:
            stable = 0
            last_h = h

    soup = BeautifulSoup(driver.page_source, "html.parser")
    links = soup.find_all("a", href=lambda h: h and "/forum?id=" in h)

    urls, seen = [], set()
    for a in links:
        href = a.get("href")
        if not href:
            continue
        if href.startswith("/"):
            href = "https://openreview.net" + href
        fid = parse_forum_id(href)
        if fid and fid not in seen:
            seen.add(fid)
            urls.append(href)
            if cap and len(urls) >= cap:
                break
    return urls

def pdf_url_for_forum(fid: str) -> str:
    # OpenReview predictable PDF endpoint
    return f"https://openreview.net/pdf?id={fid}"

def download_pdf(url: str, dest: pathlib.Path):
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)

def run_once():
    driver = make_driver()
    try:
        forum_urls = get_forum_urls(driver, cap=MAX_PAPERS)
        print(f"found {len(forum_urls)} forum links for '{SEARCH_TERM}'")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    for i, forum_url in enumerate(forum_urls, start=1):
        fid = parse_forum_id(forum_url)
        if not fid:
            print(f"[{i}] skip (no forum id): {forum_url}")
            continue

        pdf_url = pdf_url_for_forum(fid)
        outfile = DOWNLOAD_DIR / f"{fid}.pdf"

        if outfile.exists():
            print(f"[{i}] exists, skip: {outfile.name}")
            continue

        if DRY_RUN:
            print(f"[{i}] DRY_RUN would download → {pdf_url} → {outfile}")
            continue

        try:
            print(f"[{i}] downloading {fid}.pdf")
            download_pdf(pdf_url, outfile)
            print(f"[{i}] saved: {outfile}")
        except Exception as e:
            print(f"[{i}] error downloading {fid}: {e}")

if __name__ == "__main__":
    print(f"=== OpenReview PDF-only ({SEARCH_TERM=}, {MAX_PAPERS=}, {DRY_RUN=}) ===")
    run_once()
    print("done.")

