"""
ICML (International Conference on Machine Learning) Scraper

This module provides functionality to scrape research papers from ICML conference websites.
Supports both general paper scraping and topic-filtered scraping.
"""

import requests
from bs4 import BeautifulSoup
import urllib.parse
import logging

logger = logging.getLogger(__name__)

# Default headers for web scraping
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_icml_details(icml_url: str):
    """Fetch authors, abstract, date, and PDF link from ICML paper page.
    
    Args:
        icml_url: URL of the ICML paper page
        
    Returns:
        tuple: (authors, abstract, date, pdf_link)
    """
    logger.info(f"Fetching ICML details: {icml_url}")
    r = requests.get(icml_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Extract title from the main title element
    title_elem = soup.find("h2", class_="card-title main-title")
    title = title_elem.text.strip() if title_elem else ""
    
    # If not found, try alternative selectors
    if not title:
        title_elem = soup.find("title")
        if title_elem:
            title = title_elem.text.strip()
            # Remove "ICML Poster" prefix if present
            if title.startswith("ICML Poster "):
                title = title[12:]

    # Extract authors from the subtitle
    authors = []
    author_elem = soup.find("h3", class_="card-subtitle")
    if author_elem:
        author_text = author_elem.get_text().strip()
        # Split by middle dot or ampersand
        authors = [author.strip() for author in author_text.replace("·", "&").split("&")]
        authors = [author for author in authors if author]  # Remove empty strings

    # Extract abstract from the abstract section
    abstract = ""
    abstract_span = soup.find("span", class_="font-weight-bold", string="Abstract:")
    if abstract_span:
        # Get the next sibling or parent's next sibling that contains the abstract text
        abstract_container = abstract_span.parent or abstract_span
        abstract_text = abstract_container.get_text().strip()
        # Remove "Abstract:" prefix
        if abstract_text.startswith("Abstract:"):
            abstract = abstract_text[9:].strip()
    
    # If not found, try alternative selector
    if not abstract:
        abstract_div = soup.find("div", id="abstractExample")
        if abstract_div:
            abstract_text = abstract_div.get_text().strip()
            if abstract_text.startswith("Abstract:"):
                abstract = abstract_text[9:].strip()

    # Extract date from the page (ICML papers usually have conference date)
    date = None
    # Look for date information in various places
    date_elements = soup.find_all(string=lambda text: text and "2025" in text and any(month in text for month in ["Jul", "July", "Jun", "June"]))
    if date_elements:
        date = date_elements[0].strip()

    # Find OpenReview link
    openreview_link = None
    openreview_elem = soup.find("a", string=lambda text: text and "openreview" in text.lower())
    if openreview_elem and openreview_elem.get("href"):
        openreview_link = openreview_elem["href"]
        if openreview_link.startswith("/"):
            openreview_link = "https://icml.cc" + openreview_link

    # Find PDF link - ICML papers often have direct PDF links
    pdf_link = None
    pdf_elem = soup.find("a", string=lambda text: text and "pdf" in text.lower())
    if pdf_elem and pdf_elem.get("href"):
        pdf_link = pdf_elem["href"]
        if pdf_link.startswith("/"):
            pdf_link = "https://icml.cc" + pdf_link

    return authors, abstract, date, pdf_link


def extract_papers_icml(year: int, limit: int = 10, custom_url: str = None, topic_filter: str = None):
    """Scrape ICML website for papers and return paper metadata.
    
    Args:
        year: Conference year
        limit: Maximum papers to process
        custom_url: Custom URL to scrape (e.g., filtered pages)
        topic_filter: Topic filter for URL construction
        
    Returns:
        dict: Dictionary with 'uploaded', 'found', and 'papers' keys
    """
    # Build the base URL
    if custom_url:
        base = custom_url
        logger.info(f"🌐 Scraping ICML custom URL: {base}")
    elif topic_filter:
        # URL encode the topic filter
        encoded_filter = urllib.parse.quote(topic_filter)
        base = f"https://icml.cc/virtual/{year}/papers.html?filter=topic&search={encoded_filter}"
        logger.info(f"🌐 Scraping ICML filtered by topic '{topic_filter}': {base}")
    else:
        base = f"https://icml.cc/virtual/{year}/papers.html"
        logger.info(f"🌐 Scraping ICML index: {base}")
    
    try:
        r = requests.get(base, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Check if we got a valid response with papers
        if "No topics available" in r.text or "No sessions available" in r.text:
            logger.warning(f"⚠️ No papers found on filtered page. This might require authentication or cookies.")
            # Try the main papers page as fallback
            if custom_url or topic_filter:
                logger.info("🔄 Falling back to main papers page...")
                fallback_url = f"https://icml.cc/virtual/{year}/papers.html"
                r = requests.get(fallback_url, headers=HEADERS, timeout=30)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")

        # Find paper links - ICML uses different structure than ICLR
        article_links = {}
        
        # Try multiple selectors for ICML paper links
        paper_selectors = [
            "a[href*='/poster/']",  # Poster papers
            "a[href*='/oral/']",    # Oral papers  
            "a[href*='/spotlight/']", # Spotlight papers
            "a[href*='/paper/']",   # General paper links
            ".paper-title a",       # Paper title links
            ".paper-link",          # Direct paper links
            "a[href*='/virtual/']"  # Virtual conference links
        ]
        
        for selector in paper_selectors:
            links = soup.select(selector)
            for link in links:
                if link.get("href") and link.text.strip():
                    title = link.text.strip()
                    href = link["href"]
                    url = "https://icml.cc" + href if href.startswith("/") else href
                    # Only include valid paper URLs
                    if any(path in url for path in ['/poster/', '/oral/', '/spotlight/', '/paper/']):
                        article_links[title] = url

        # If no papers found with selectors, try to find any links that might be papers
        if not article_links:
            logger.info("🔍 No papers found with standard selectors, trying broader search...")
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                href = link.get("href", "")
                if href and any(path in href for path in ['/poster/', '/oral/', '/spotlight/', '/paper/']):
                    title = link.text.strip()
                    if title and len(title) > 10:  # Reasonable title length
                        url = "https://icml.cc" + href if href.startswith("/") else href
                        article_links[title] = url

        logger.info(f"Found {len(article_links)} papers on ICML site.")

        # Process papers up to the limit
        processed_papers = []
        processed = 0
        
        for i, (title, paper_url) in enumerate(article_links.items(), start=1):
            if processed >= limit:
                logger.info(f"Limit of {limit} papers reached, stopping early.")
                break

            try:
                logger.info(f"[{i}/{len(article_links)}] Processing: {title}")
                
                # Fetch paper details from ICML page
                authors, abstract, date, pdf_link = fetch_icml_details(paper_url)
                
                paper_data = {
                    "title": title,
                    "url": paper_url,
                    "authors": authors,
                    "abstract": abstract,
                    "date": date,
                    "pdf_link": pdf_link
                }
                
                processed_papers.append(paper_data)
                processed += 1

            except Exception as e:
                logger.exception(f"❌ Error processing {title}: {e}")
                continue

        logger.info(f"Done. Processed {processed} papers.")
        return {
            "uploaded": processed, 
            "found": len(article_links),
            "papers": processed_papers
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to scrape ICML {year}: {e}")
        return {"uploaded": 0, "found": 0, "papers": []}


def get_icml_papers_by_topic(year: int, topic: str, limit: int = 10):
    """Convenience function to get ICML papers filtered by topic.
    
    Args:
        year: Conference year
        topic: Topic to filter by (e.g., "Deep Learning->Large Language Models")
        limit: Maximum papers to return
        
    Returns:
        dict: Dictionary with paper metadata
    """
    return extract_papers_icml(year=year, limit=limit, topic_filter=topic)


def get_icml_papers_from_url(url: str, limit: int = 10):
    """Convenience function to get ICML papers from a specific URL.
    
    Args:
        url: Custom ICML URL to scrape
        limit: Maximum papers to return
        
    Returns:
        dict: Dictionary with paper metadata
    """
    return extract_papers_icml(custom_url=url, limit=limit)
