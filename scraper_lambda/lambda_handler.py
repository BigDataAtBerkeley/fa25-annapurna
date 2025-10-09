import os
import logging
from sources.iclr_scraper import extract_papers_iclr
from sources.icml_scraper import extract_papers_icml
from sources.arxiv_scraper import extract_papers_arxiv
from sources.neurips_scraper import extract_papers_neurips
from sources.openreview_scraper import extract_papers_openreview
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def lambda_handler(event, context):
    source = os.getenv("SOURCE", "iclr").lower()
    year = int(os.getenv("SCRAPE_YEAR", "2025"))
    max_papers = int(os.getenv("MAX_PAPERS", "3"))

    logger.info(f"Starting Lambda for source={source}, year={year}, limit={max_papers}")

    if source == "iclr":
        result = extract_papers_iclr(year, limit=max_papers)
    elif source == "icml":
        result = extract_papers_icml(year, limit=max_papers)
    elif source == "arxiv":
        result = extract_papers_arxiv(limit=max_papers)
    elif source == "neurips":
        result = extract_papers_neurips(limit=max_papers)
    elif source == "openreview":
        result = extract_papers_openreview(limit=max_papers)
    else:
        raise ValueError(f"Unknown source: {source}")

    logger.info(f"Lambda for {source} completed: {result}")
    return result
