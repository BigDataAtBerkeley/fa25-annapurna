from utils.logging_utils import setup_logger
from utils.request_utils import get_soup, safe_filename, _fetch_openreview_details, extract_papers_function
from utils.aws_utils import upload_pdf_to_s3, send_to_sqs
import requests

logger = setup_logger("neurips_scraper")
extract_papers_neurips = extract_papers_function("neurips", logger)
        
"""
def extract_papers_neurips(year: int, limit: int = 10, search_term: str = "LLM"):
    base = f"https://neurips.cc/virtual/{year}/papers.html?search=LLM"
    logger.info(f"Scraping NeurIPS papers from {base}")
    neurips = get_soup(base)
    cards = neurips.find_all('li', class_=False)
    paper_cards = {card.find("a").text.strip(): "https://neurips.cc" + card.find("a")['href'] for card in cards if card.find("a") and card.find("a").text and card.find("a")['href']}
    
    count = 0
    for i, (title, link) in enumerate(paper_cards.items(), start=1):
        if count >= limit:
            break
        logger.info(f"[{i}] {title}")
        try:
            psoup = get_soup(link)
            openreview_link = psoup.find("a", {"title": "OpenReview"})
            if not openreview_link:
                continue
            url = openreview_link["href"]
            if url.startswith("/"): 
                continue

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
    return {"uploaded": count, "found": len(paper_cards)}
"""