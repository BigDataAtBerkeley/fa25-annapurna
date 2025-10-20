"""
ICML (International Conference on Machine Learning) Scraper

Scrapes ICML virtual conference papers, extracts metadata,
downloads PDFs, uploads them to S3, and sends metadata to SQS.
"""

from utils.logging_utils import setup_logger
from utils.request_utils import get_soup, safe_filename
from utils.aws_utils import upload_pdf_to_s3, send_to_sqs
import requests
from bs4 import BeautifulSoup
import urllib.parse

logger = setup_logger("icml_scraper")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _fetch_icml_details(icml_url: str):
    """Fetch authors, abstract, date, and PDF link from an ICML paper page."""
    logger.info(f"Fetching ICML details: {icml_url}")
    r = requests.get(icml_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Authors
    author_elem = soup.find("h3", class_="card-subtitle")
    authors = []
    if author_elem:
        text = author_elem.get_text().strip()
        authors = [a.strip() for a in text.replace("·", "&").split("&") if a.strip()]

    # Abstract
    abstract = ""
    abs_span = soup.find("span", class_="font-weight-bold", string="Abstract:")
    if abs_span:
        abs_text = abs_span.parent.get_text().strip()
        if abs_text.startswith("Abstract:"):
            abstract = abs_text[9:].strip()
    else:
        alt_abs = soup.find("div", id="abstractExample")
        if alt_abs:
            text = alt_abs.get_text().strip()
            if text.startswith("Abstract:"):
                abstract = text[9:].strip()

    # Date
    date = None
    date_elements = soup.find_all(
        string=lambda t: t and "2025" in t and any(m in t for m in ["Jul", "July", "Jun", "June"])
    )
    if date_elements:
        date = date_elements[0].strip()

    # PDF link
    pdf_link = None
    pdf_elem = soup.find("a", string=lambda t: t and "pdf" in t.lower())
    if pdf_elem and pdf_elem.get("href"):
        href = pdf_elem["href"]
        pdf_link = "https://icml.cc" + href if href.startswith("/") else href

    return authors, abstract, date, pdf_link


def extract_papers_icml(year: int, limit: int = 10, topic_filter: str = None):
    """Scrape ICML website for papers and upload to AWS."""
    if topic_filter:
        encoded = urllib.parse.quote(topic_filter)
        base = f"https://icml.cc/virtual/{year}/papers.html?filter=topic&search={encoded}"
        logger.info(f"Scraping ICML papers filtered by topic '{topic_filter}': {base}")
    else:
        base = f"https://icml.cc/virtual/{year}/papers.html"
        logger.info(f"Scraping ICML papers from {base}")

    try:
        soup = get_soup(base)
        article_links = {}

        selectors = [
            "a[href*='/poster/']", "a[href*='/oral/']",
            "a[href*='/spotlight/']", "a[href*='/paper/']"
        ]
        for sel in selectors:
            for link in soup.select(sel):
                if link.get("href") and link.text.strip():
                    title = link.text.strip()
                    href = link["href"]
                    url = "https://icml.cc" + href if href.startswith("/") else href
                    article_links[title] = url

        logger.info(f"Found {len(article_links)} papers on ICML site.")
        count = 0

        for i, (title, url) in enumerate(article_links.items(), start=1):
            if count >= limit:
                logger.info(f"Limit of {limit} papers reached, stopping early.")
                break

            try:
                logger.info(f"[{i}/{len(article_links)}] {title}")
                authors, abstract, date, pdf_link = _fetch_icml_details(url)
                if not pdf_link:
                    logger.warning(f"No PDF found for {title}")
                    continue

                key = safe_filename(title)
                with requests.get(pdf_link, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    upload_pdf_to_s3(r.raw, key)
                send_to_sqs(title, authors, date, abstract, key)
                count += 1

            except Exception as e:
                logger.exception(f"Error processing {title}: {e}")
                continue

        logger.info(f"Done. Uploaded and queued {count} papers.")
        return {"uploaded": count, "found": len(article_links)}

    except Exception as e:
        logger.exception(f"Failed to scrape ICML {year}: {e}")
        return {"uploaded": 0, "found": 0}
