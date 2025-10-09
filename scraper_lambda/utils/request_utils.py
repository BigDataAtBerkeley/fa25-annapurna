import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def get_soup(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    from bs4 import BeautifulSoup
    return BeautifulSoup(r.text, "html.parser")

def safe_filename(title: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in title) + ".pdf"
