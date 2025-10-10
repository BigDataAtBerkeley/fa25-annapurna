# pip install requests beautifulsoup4 playwright
# playwright install

import re
import json
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def fetch_html(url):
    """Try requests first; fall back to Playwright if empty/JS-heavy page."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchScraper/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200 and len(r.text) > 1000:
            return r.text
    except Exception:
        pass

    # fallback: headless browser
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)
        html = page.content()
        browser.close()
    return html


def extract_info(url, html):
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "title": None,
        "authors": None,
        "year": None,
        "abstract": None,
        "url": url,
        "pdf_link": None
    }

    # ---- TITLE ----
    title = soup.find(["h1", "h2"], string=True)
    if not title:
        meta_title = soup.find("meta", {"property": "og:title"}) or soup.find("meta", {"name": "citation_title"})
        if meta_title:
            title = meta_title.get("content")
    result["title"] = title.get_text(strip=True) if hasattr(title, "get_text") else title

    # ---- AUTHORS ----
    authors = []
    for tag in soup.find_all("meta", {"name": "citation_author"}):
        if tag.get("content"):
            authors.append(tag["content"].strip())
    if not authors:
        author_divs = soup.find_all(["div", "span", "p"], class_=re.compile("author", re.I))
        for div in author_divs:
            text = div.get_text(" ", strip=True)
            if text and len(text.split()) < 8:
                authors.append(text)
    result["authors"] = ", ".join(authors) if authors else None

    # ---- YEAR ----
    year = None
    meta_year = soup.find("meta", {"name": "citation_year"}) or soup.find("meta", {"name": "citation_date"})
    if meta_year:
        content = meta_year.get("content")
        if content and re.search(r"\d{4}", content):
            year = int(re.search(r"\d{4}", content).group())
    if not year:
        text = soup.get_text()
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            year = int(match.group())
    result["year"] = year

    # ---- ABSTRACT ----
    abstract = None
    meta_desc = soup.find("meta", {"name": "description"}) or soup.find("meta", {"name": "citation_abstract"})
    if meta_desc:
        abstract = meta_desc.get("content")
    if not abstract:
        abs_block = soup.find("div", class_=re.compile("abstract", re.I)) \
                  or soup.find("p", class_=re.compile("abstract", re.I))
        if abs_block:
            abstract = abs_block.get_text(" ", strip=True)
    result["abstract"] = abstract

    # ---- PDF LINK ----
    pdf = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".pdf") or "pdf" in href.lower():
            pdf = href if href.startswith("http") else requests.compat.urljoin(url, href)
            break
    result["pdf_link"] = pdf

    return result


def scrape_paper(url):
    html = fetch_html(url)
    data = extract_info(url, html)
    return data


if __name__ == "__main__":
    urls = [
        "https://openreview.net/forum?id=d0di5jC9kb",
        "https://openreview.net/forum?id=8qqMeF9EmT",
        "https://openreview.net/forum?id=63faJh4DpZ"
    ]

    all_results = []

    for url in urls:
        print(f"Scraping: {url}")
        try:
            info = scrape_paper(url)
            all_results.append(info)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    # ---- Save to JSON ----
    with open("openreview_papers.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n✅ Done! Results saved to openreview_papers.json")
