import requests
from utils.aws_utils import upload_pdf_to_s3, send_to_sqs

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

URL_PREFIXES = {
        "neurips": "https://neurips.cc",
        "mlsys": "https://mlsys.org",
        "iclr": "https://iclr.cc"
}

def get_soup(url: str):
    # Fetching and parsing HTML content from a URL
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    from bs4 import BeautifulSoup
    return BeautifulSoup(r.text, "html.parser")

def safe_filename(title: str) -> str:
    # Sanitizing title to create a safe filename
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in title) + ".pdf"

def _fetch_openreview_details(url: str):
    # Parsing OpenReview page for metadata and PDF link
    soup = get_soup(url)
    authors = [m["content"] for m in soup.find_all("meta", {"name": "citation_author"})]
    abstract = soup.find("meta", {"name": "citation_abstract"})
    date = soup.find("meta", {"name": "citation_online_date"})
    pdf = soup.find("a", {"title": "Download PDF"})
    pdf_link = "https://openreview.net" + pdf["href"] if pdf and pdf.get("href") else None
    return authors, abstract["content"] if abstract else "", date["content"] if date else "", pdf_link

def extract_papers_function(source: str, logger, search_term: str = "LLM"):
    # Returns a function to extract papers from a specified source
    def func(year: int, limit: int = 10):
        # Verifying source
        if source not in URL_PREFIXES:
            logger.error(f"Unknown source: {source}")
            return {"uploaded": 0, "found": 0}
        
        # Retrieving papers from the source
        prefix = URL_PREFIXES[source]
        base = f"{prefix}/virtual/{year}/papers.html?search={search_term}"
        logger.info(f"Scraping {source} papers from {base}")
        bsoup = get_soup(base)
        cards = bsoup.find_all('li', class_=False)
        paper_cards = {card.find("a").text.strip(): prefix + card.find("a")['href'] for card in cards if card.find("a") and card.find("a").text and card.find("a")['href']}
        
        # Pulling documents from OpenReview and processing
        count = 0
        for i, (title, link) in enumerate(paper_cards.items(), start=1):
            if count >= limit:
                break
            logger.info(f"[{i}] {title}: {link}")
            count += 1
            try:
                psoup = get_soup(link)
                openreview_link = psoup.find("a", {"title": "OpenReview"})
                if not openreview_link:
                    logger.exception(f"Error processing {title}: could not find a link to OpenReview.")
                    continue
                url = openreview_link["href"]
                if url.startswith("/"): 
                    logger.exception(f"Error processing {title}: OpenReview link references site.")
                    continue

                authors, abstract, date, pdf_link = _fetch_openreview_details(url)
                if not pdf_link:
                    logger.exception(f"Error processing {title}: could not find PDF link on OpenReview.")
                    continue
                
                key = safe_filename(title)
                with requests.get(pdf_link, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    upload_pdf_to_s3(r.raw, key)
                send_to_sqs(title, authors, date, abstract, key)
            except Exception as e:
                logger.exception(f"Error processing {title}: {e}")
                continue
        logger.info(f"Done. Uploaded and queued {count} papers.")
        return {"uploaded": count, "found": len(paper_cards)}
    return func