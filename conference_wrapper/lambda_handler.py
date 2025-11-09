# counter_lambda.py
import requests
from bs4 import BeautifulSoup
from utils.logging_utils import setup_logger

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

logger = setup_logger(__name__)

def retrive_batches(source: str, year: int, batch_size: int =30, search_term: str ="LLM", test_count: int =None):
    
    # Get paper count
    url = f"https://{source}.cc/virtual/{year}/papers.html?search={search_term}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    cards = soup.find_all('li', class_=False)
    paper_cards = [card for card in cards if card.find("a") and card.find("a").text and card.find("a")['href']]
    total_papers = test_count if test_count else len(paper_cards)
    
    # Create batches
    batches = []
    for i in range(0, total_papers, batch_size):
        batches.append({
            "source": source,
            "year": year,
            "search_term": search_term,
            "start_index": i + 1,  # 1-indexed
            "end_index": min(i + batch_size, total_papers),
            "limit": batch_size  # Still respect limit
        })
    
    return {
        "total_papers": total_papers,
        "batch_size": batch_size,
        "num_batches": len(batches),
        "batches": batches
    }
    
def lambda_handler(event, context):
    
    SOURCE = event.get('source', 'iclr')
    YEAR = int(event.get('year', 2025))
    BATCH_SIZE = int(event.get('batch_size', 30))
    SEARCH_TERM = event.get('search_term', 'LLM')
    TEST_COUNT = int(event.get('test_count', None))
    
    if SOURCE not in ['iclr', 'mlsys', 'neurips', 'icml']:
        raise ValueError(f"Unknown source: {SOURCE}. This MapState execution is for conferences only. Supported: iclr, mlsys, neurips, icml")

    result = retrive_batches(SOURCE, YEAR, BATCH_SIZE, SEARCH_TERM, TEST_COUNT)
    logger.info(f"Generated {result['num_batches']} batches for source={SOURCE}, year={YEAR}, batch_size={BATCH_SIZE}")
    return result