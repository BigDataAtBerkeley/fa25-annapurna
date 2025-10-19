"""
ACL (Association for Computational Linguistics) Scraper

This module provides functionality to scrape research papers from ACL Anthology,
specifically focusing on ACL 2025 conference papers related to LLMs.
"""

import requests
from bs4 import BeautifulSoup
import urllib.parse
import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Default headers for web scraping
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_acl_paper_details(acl_url: str):
    """Fetch authors, abstract, date, and PDF link from ACL Anthology paper page.
    
    Note: ACL Anthology structure is different - papers are often in volume pages
    with direct PDF links rather than individual paper detail pages.
    
    Args:
        acl_url: URL of the ACL Anthology paper page
        
    Returns:
        tuple: (authors, abstract, date, pdf_link)
    """
    logger.info(f"Fetching ACL paper details: {acl_url}")
    
    try:
        r = requests.get(acl_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        
        # Check if we got a PDF directly
        if r.headers.get('content-type', '').startswith('application/pdf'):
            logger.info("Received PDF directly from ACL URL")
            return [], "", "2025", acl_url
        
        soup = BeautifulSoup(r.text, "html.parser")

        # Extract title from page
        title = ""
        title_elem = soup.find("h1", {"id": "title"}) or soup.find("h1", class_="title") or soup.find("title")
        if title_elem:
            title = title_elem.get_text().strip()

        # Extract authors
        authors = []
        author_elems = soup.find_all("a", href=re.compile(r"/people/"))
        for author_elem in author_elems:
            author_name = author_elem.get_text().strip()
            if author_name and author_name not in authors:
                authors.append(author_name)

        # Extract abstract
        abstract = ""
        abstract_elem = soup.find("div", {"id": "abstract"}) or soup.find("div", class_="abstract")
        if abstract_elem:
            abstract = abstract_elem.get_text().strip()

        # Extract date
        date = "2025"  # Default for ACL 2025

        # Find PDF link
        pdf_link = None
        pdf_elem = soup.find("a", string=re.compile(r"PDF", re.I)) or soup.find("a", href=re.compile(r"\.pdf$"))
        if pdf_elem and pdf_elem.get("href"):
            pdf_link = pdf_elem["href"]
            if pdf_link.startswith("/"):
                pdf_link = "https://aclanthology.org" + pdf_link

        return authors, abstract, date, pdf_link

    except Exception as e:
        logger.error(f"Error fetching ACL paper details from {acl_url}: {e}")
        return [], "", None, None


def extract_acl_papers(year: int = 2025, limit: int = 10, keyword_filter: str = "LLM"):
    """Scrape ACL Anthology for papers related to LLMs.
    
    ACL Anthology has a different structure - it organizes papers in volumes
    rather than individual pages. We'll look for direct PDF links.
    
    Args:
        year: Conference year
        limit: Maximum papers to process
        keyword_filter: Keywords to filter papers (default: "LLM")
        
    Returns:
        dict: Dictionary with 'uploaded', 'found', and 'papers' keys
    """
    # ACL Anthology URL for ACL 2025
    base_url = f"https://aclanthology.org/events/acl-{year}/"
    
    logger.info(f"🌐 Scraping ACL {year} papers from: {base_url}")
    
    try:
        r = requests.get(base_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Find volume links (ACL organizes papers in volumes like long, short, demo)
        volume_links = []
        
        # Look for volume links
        volume_selectors = [
            "a[href*='/volumes/2025.acl-']",  # ACL 2025 volume links
            "a[href*='/volumes/']"            # General volume links
        ]
        
        for selector in volume_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get("href", "")
                if href and "volumes" in href and "2025.acl-" in href:
                    if href.startswith("/"):
                        full_url = "https://aclanthology.org" + href
                    else:
                        full_url = href
                    volume_links.append(full_url)

        logger.info(f"Found {len(volume_links)} ACL {year} volumes: {volume_links}")

        # Process each volume to find papers
        all_papers = []
        
        for volume_url in volume_links[:3]:  # Limit to first 3 volumes for now
            try:
                logger.info(f"📚 Processing volume: {volume_url}")
                vol_r = requests.get(volume_url, headers=HEADERS, timeout=30)
                vol_r.raise_for_status()
                vol_soup = BeautifulSoup(vol_r.text, "html.parser")
                
                # Look for individual paper links in the volume
                paper_selectors = [
                    "a[href*='/2025.acl-long.']",  # Long papers
                    "a[href*='/2025.acl-short.']", # Short papers
                    "a[href*='/2025.acl-demo.']",  # Demo papers
                    "a[href*='.pdf']",             # Direct PDF links
                ]
                
                for selector in paper_selectors:
                    links = vol_soup.select(selector)
                    for link in links:
                        href = link.get("href", "")
                        title = link.get_text().strip()
                        
                        if href and title and len(title) > 10:
                            # Skip non-paper links
                            if any(skip in title.lower() for skip in ["pdf", "bib", "proceedings", "volume"]):
                                continue
                                
                            # Construct full URL
                            if href.startswith("/"):
                                full_url = "https://aclanthology.org" + href
                            else:
                                full_url = href
                            
                            all_papers.append({
                                "title": title,
                                "url": full_url
                            })
                
            except Exception as e:
                logger.warning(f"❌ Error processing volume {volume_url}: {e}")
                continue

        logger.info(f"Found {len(all_papers)} total papers across volumes.")

        # Filter papers by keywords and process them
        filtered_papers = []
        processed = 0
        
        llm_keywords = [
            # Core LLM terms
            "llm", "large language model", "language model", "foundation model", "pretrained model",
            "neural language model", "generative language model",
            
            # Specific models
            "gpt", "bert", "t5", "bart", "roberta", "deberta", "chatgpt", "claude", "gemini",
            "llama", "alpaca", "vicuna", "mistral", "falcon", "palm", "chinchilla", "gopher",
            
            # Architecture terms
            "transformer", "attention mechanism", "self-attention", "multi-head attention",
            
            # Generation terms
            "generative", "text generation", "language generation", "text-to-text",
            "conditional generation", "controllable generation",
            
            # NLP terms
            "natural language", "nlp", "natural language processing", "natural language understanding",
            "natural language generation", "dialogue system", "conversational ai", "chatbot",
            "question answering", "text summarization", "machine translation", "sentiment analysis",
            
            # AI/ML terms
            "artificial intelligence", "machine learning", "deep learning", "neural network",
            "fine-tuning", "prompting", "prompt engineering", "in-context learning",
            "few-shot learning", "zero-shot learning", "chain-of-thought", "instruction tuning",
            
            # Application terms
            "code generation", "code completion", "program synthesis", "reasoning", "commonsense",
            "knowledge extraction", "information extraction", "text mining", "content creation"
        ]
        
        for paper_info in all_papers:
            if processed >= limit:
                break
                
            try:
                title = paper_info["title"]
                paper_url = paper_info["url"]
                
                # Check if title contains LLM-related keywords
                title_lower = title.lower()
                if not any(keyword in title_lower for keyword in llm_keywords):
                    continue
                
                logger.info(f"[{processed + 1}/{limit}] Processing: {title}")
                
                # For ACL, we'll try to get the PDF directly
                # Many ACL papers have direct PDF links
                pdf_link = paper_url
                if not paper_url.endswith('.pdf'):
                    # Try to construct PDF URL
                    if '/2025.acl-' in paper_url:
                        pdf_link = paper_url + '.pdf'
                
                paper_data = {
                    "title": title,
                    "url": paper_url,
                    "authors": [],  # ACL structure makes author extraction difficult
                    "abstract": "",  # ACL structure makes abstract extraction difficult
                    "date": "2025",
                    "pdf_link": pdf_link
                }
                
                filtered_papers.append(paper_data)
                processed += 1

            except Exception as e:
                logger.exception(f"❌ Error processing {title}: {e}")
                continue

        logger.info(f"Done. Found {processed} LLM-related papers.")
        return {
            "uploaded": processed, 
            "found": len(all_papers),
            "papers": filtered_papers
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to scrape ACL {year}: {e}")
        return {"uploaded": 0, "found": 0, "papers": []}


def get_acl_llm_papers(year: int = 2025, limit: int = 10):
    """Convenience function to get ACL papers related to LLMs.
    
    Args:
        year: Conference year
        limit: Maximum papers to return
        
    Returns:
        dict: Dictionary with paper metadata
    """
    return extract_acl_papers(year=year, limit=limit, keyword_filter="LLM")


def search_acl_papers_by_keywords(year: int = 2025, keywords: List[str] = None, limit: int = 10):
    """Search ACL papers by specific keywords.
    
    Args:
        year: Conference year
        keywords: List of keywords to search for
        limit: Maximum papers to return
        
    Returns:
        dict: Dictionary with paper metadata
    """
    if keywords is None:
        keywords = ["LLM", "large language model"]
    
    # Convert keywords list to a single filter string
    keyword_filter = " ".join(keywords)
    return extract_acl_papers(year=year, limit=limit, keyword_filter=keyword_filter)
