from utils.logging_utils import setup_logger
from utils.aws_utils import upload_pdf_to_s3, send_to_sqs
from utils.request_utils import safe_filename

import arxiv
import datetime

logger = setup_logger("arxiv_scraper")

def extract_papers_arxiv(limit: int = 10):
    query = '(LLM OR "large language model" OR "large language models") AND (cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:stat.ML)'
    logger.info(f"Querying arXiv with: {query}")
    client = arxiv.Client(num_retries=5, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    count = 0
    found = 0

    for i, result in enumerate(client.results(search), start=1):
        found += 1
        if count >= limit:
            break

        title = result.title.strip()
        authors = [a.name for a in result.authors]
        abstract = result.summary.strip()
        date = result.published.isoformat() if result.published else str(datetime.date.today())
        pdf_link = result.pdf_url
        arxiv_url = result.entry_id

        logger.info(f"[{i}] {title}")
        logger.debug(f"arXiv URL: {arxiv_url}")
        logger.debug(f"PDF link: {pdf_link}")

        if not pdf_link:
            logger.warning(f"No PDF found for {title}")
            continue

        try:
            key = safe_filename(title)
            pdf_stream = arxiv.Client()._session.get(pdf_link, stream=True, timeout=60)
            pdf_stream.raise_for_status()

            upload_pdf_to_s3(pdf_stream.raw, key)
            send_to_sqs(title, authors, date, abstract, key)
            count += 1

        except Exception as e:
            logger.exception(f"Error processing {title}: {e}")
            continue

    logger.info(f"Done. Uploaded and queued {count} of {found} papers.")
    return {"uploaded": count, "found": found}
