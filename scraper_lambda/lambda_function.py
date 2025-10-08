import os
import json
import uuid
import logging
import requests
from bs4 import BeautifulSoup
import boto3
from datetime import datetime

# --- Logging Setup ---
LOG_DIR = "/tmp/logs"  # /tmp is writable in AWS Lambda
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "scraper.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# File + console logging
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Scraper Lambda initialized")

# --- Environment & AWS setup ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("BUCKET_NAME")
QUEUE_URL = os.getenv("QUEUE_URL")

session = boto3.Session(region_name=AWS_REGION)
s3 = session.client("s3")
sqs = session.client("sqs")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# --- Utility Functions ---
def _safe_filename(title: str) -> str:
    """Create a safe filename for S3 upload."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in title) + ".pdf"

def _send_message(title, authors, date, abstract, s3_bucket, s3_key):
    """Send paper metadata to SQS."""
    msg = {
        "title": title,
        "authors": authors,
        "date": date,
        "abstract": abstract,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
    }
    resp = sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(msg),
        MessageGroupId="research-parsing",
        MessageDeduplicationId=str(uuid.uuid4()),
    )
    logger.info(f"Sent to SQS | MessageId={resp['MessageId']} | Title='{title}'") ## logging when a paper is sent to SQS

def _fetch_openreview_details(openreview_url: str):
    """Fetch authors, abstract, date, and PDF link from OpenReview."""
    logger.info(f"Fetching OpenReview details: {openreview_url}")
    r = requests.get(openreview_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    authors = [m["content"] for m in soup.find_all("meta", attrs={"name": "citation_author"})]
    abstract_meta = soup.find("meta", attrs={"name": "citation_abstract"})
    date_meta = soup.find("meta", attrs={"name": "citation_online_date"})
    pdf_a = soup.find("a", attrs={"title": "Download PDF"})

    abstract = abstract_meta["content"] if abstract_meta else ""
    date = date_meta["content"] if date_meta else None
    pdf_link = None
    if pdf_a and pdf_a.get("href"):
        href = pdf_a["href"]
        pdf_link = "https://openreview.net" + href if href.startswith("/") else href

    return authors, abstract, date, pdf_link

def _fetch_pdf_and_upload(pdf_url: str, key: str):
    """Download paper PDF and upload to S3."""
    logger.info(f"⬇️ Downloading PDF: {pdf_url}")
    with requests.get(pdf_url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        s3.upload_fileobj(r.raw, BUCKET, key)
    logger.info(f"Uploaded PDF to s3://{BUCKET}/{key}")

# --- Main Scraper ---
def extract_papers_iclr(year: int, limit: int = 10):
    """Scrape ICLR website for papers and push to S3 + SQS."""
    base = f"https://iclr.cc/virtual/{year}/papers.html"
    logger.info(f"🌐 Scraping ICLR index: {base}")
    r = requests.get(base, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    cards = soup.find_all("li", class_=False)
    article_links = {}
    for c in cards:
        a = c.find("a")
        if a and a.get("href") and a.text.strip():
            title = a.text.strip()
            href = a["href"]
            url = "https://iclr.cc" + href if href.startswith("/") else href
            article_links[title] = url

    logger.info(f"Found {len(article_links)} papers on ICLR site.")

    processed = 0
    for i, (title, paper_url) in enumerate(article_links.items(), start=1):
        if processed >= limit:
            logger.info(f"Limit of {limit} papers reached, stopping early.")
            break

        try:
            logger.info(f"[{i}/{len(article_links)}] Processing: {title}")
            pr = requests.get(paper_url, headers=HEADERS, timeout=30)
            pr.raise_for_status()
            psoup = BeautifulSoup(pr.text, "html.parser")
            a = psoup.find("a", {"title": "OpenReview"})
            if not a or not a.get("href"):
                logger.warning(f"⚠️ No OpenReview link for {title}. Skipping.")
                continue

            openreview_url = a["href"]
            if openreview_url.startswith("/"):
                openreview_url = "https://iclr.cc" + openreview_url

            authors, abstract, date, pdf_link = _fetch_openreview_details(openreview_url)
            if not pdf_link:
                logger.warning(f"⚠️ No PDF link for {title}. Skipping.")
                continue

            key = _safe_filename(title)
            _fetch_pdf_and_upload(pdf_link, key)
            _send_message(title, authors, date, abstract, BUCKET, key)
            processed += 1

        except Exception as e:
            logger.exception(f"❌ Error processing {title}: {e}")
            continue

    logger.info(f"Done. Uploaded and enqueued {processed} papers.")
    return {"uploaded": processed, "found": len(article_links)}

# --- Lambda Handler ---
def lambda_handler(event, context):
    year = int(os.getenv("ICLR_YEAR", "2025"))
    max_papers = int(os.getenv("MAX_PAPERS", "3"))  # Limiting max papers to 3 for now  (can change this in .env)
    if not BUCKET or not QUEUE_URL:
        raise RuntimeError("BUCKET_NAME and QUEUE_URL env vars are required")

    start = datetime.utcnow()
    result = extract_papers_iclr(year, limit=max_papers)
    logger.info(f"Lambda execution completed in {datetime.utcnow() - start}. Summary: {result}")
    return result
