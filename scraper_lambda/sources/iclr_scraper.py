from utils.logging_utils import setup_logger
from utils.request_utils import get_soup, safe_filename
from utils.aws_utils import upload_pdf_to_s3, send_to_sqs
import requests

logger = setup_logger("iclr_scraper")

def extract_papers_iclr(year: int, limit: int = 10):
    base = f"https://iclr.cc/virtual/{year}/papers.html"
    logger.info(f"Scraping ICLR papers from {base}")
    soup = get_soup(base)

    cards = soup.find_all("li", class_=False)
    article_links = {
        a.text.strip(): "https://iclr.cc" + a["href"]
        for c in cards if (a := c.find("a")) and a.get("href")
    }

    count = 0
    for i, (title, link) in enumerate(article_links.items(), start=1):
        if count >= limit:
            break
        logger.info(f"[{i}] {title}")
        try:
            psoup = get_soup(link)
            openreview_link = psoup.find("a", {"title": "OpenReview"})
            if not openreview_link:
                continue
            url = openreview_link["href"]
            if url.startswith("/"): url = "https://iclr.cc" + url

            authors, abstract, date, pdf_link = _fetch_openreview_details(url)
            if not pdf_link:
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

def _fetch_openreview_details(url: str):
    soup = get_soup(url)
    authors = [m["content"] for m in soup.find_all("meta", {"name": "citation_author"})]
    abstract = soup.find("meta", {"name": "citation_abstract"})
    date = soup.find("meta", {"name": "citation_online_date"})
    pdf = soup.find("a", {"title": "Download PDF"})
    pdf_link = "https://openreview.net" + pdf["href"] if pdf and pdf.get("href") else None
    return authors, abstract["content"] if abstract else "", date["content"] if date else "", pdf_link
