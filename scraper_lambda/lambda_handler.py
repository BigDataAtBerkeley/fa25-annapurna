import os
import logging
from sources.iclr_scraper import extract_papers_iclr
from sources.arxiv_scraper import extract_papers_arxiv
from sources.icml_scraper import extract_papers_icml
from sources.mlsys_scraper import extract_papers_mlsys
from sources.neurips_scraper import extract_papers_neurips
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def lambda_handler(event, context):
    source = event.get("source", os.getenv("SOURCE", "iclr")).lower()
    year = int(event.get("year", os.getenv("SCRAPE_YEAR", "2025")))
    max_papers = int(event.get("batch_size", os.getenv("MAX_PAPERS", "30")))
    
    start_index = int(event.get("start_index", 1))
    end_index = event.get("end_index")  # Can be None
    if end_index is not None:
        end_index = int(end_index)
    
    search_term = event.get("search_term", "LLM")

    logger.info(f"Starting Lambda for source={source}, year={year}, limit={max_papers}, "
                f"batch: {start_index}-{end_index if end_index else 'end'}")

    if source == "iclr":
        result = extract_papers_iclr(year, limit=max_papers, start_index=start_index, end_index=end_index, search_term=search_term)
    elif source == "mlsys":
        result = extract_papers_mlsys(year, limit=max_papers, start_index=start_index, end_index=end_index, search_term=search_term)
    elif source == "neurips":
        result = extract_papers_neurips(year, limit=max_papers, start_index=start_index, end_index=end_index, search_term=search_term)
    elif source == "arxiv":
        # ArXiv Scraper gets updated daily - do not invoke via MapState. Cron job only for PaperScraperArXiv only.
        result = extract_papers_arxiv(limit=max_papers)
    elif source == 'icml':
        result = extract_papers_icml(year, limit=max_papers, start_index=start_index, end_index=end_index, search_term=search_term)
    else:
        raise ValueError(f"Unknown source: {source}. Supported: iclr, mlsys, neurips, icml, arxiv")

    logger.info(f"Lambda for {source} completed batch {start_index}-{end_index}: {result}")
    return result