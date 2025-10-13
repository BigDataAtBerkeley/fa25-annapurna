from utils.logging_utils import setup_logger
from utils.request_utils import get_soup, safe_filename, _fetch_openreview_details, extract_papers_function
from utils.aws_utils import upload_pdf_to_s3, send_to_sqs
import requests

logger = setup_logger("mylsys_scraper")
extract_papers_mlsys = extract_papers_function("mlsys", logger)

"""
def extract_papers_mlsys(year: int, limit: int = 10):
    base = f"https://mlsys.org/virtual/{year}/papers.html"
    logger.info(f"Scraping MLSYS papers from {base}")
    mlsys = get_soup(base)
    cards = mlsys.find_all('li', class_=False)
    paper_cards = {card.find('a').text.strip(): "https://mlsys.org" + card.find('a')['href'] for card in cards if card.find('a') and card.find('a').text and card.find('a')['href']}
    
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