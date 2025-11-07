# scraper_lambda/openreview_scraper.py
import os, re, json, uuid, datetime, requests
from typing import List
import boto3
from bs4 import BeautifulSoup

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
QUEUE_URL = os.getenv("QUEUE_URL")
BUCKET_NAME = os.getenv("BUCKET_NAME")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")

sqs = boto3.client(
    "sqs",
    region_name=AWS_REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY,
)
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY,
)

def _sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^\w\s.-]", "_", name)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return (clean[:180] + "_" + str(uuid.uuid4())[:8] if len(clean) > 190 else clean)

def _send_message(title: str, authors: List[str], date: str, abstract: str, s3_bucket: str, s3_key: str):
    body = {
        "title": title,
        "authors": authors,
        "date": date,
        "abstract": abstract,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
    }
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(body),
        MessageGroupId="research-parsing",
        MessageDeduplicationId=str(uuid.uuid4()),
    )

def _extract_meta(soup: BeautifulSoup, name: str, default=None):
    tag = soup.find("meta", attrs={"name": name})
    return tag["content"].strip() if tag and tag.has_attr("content") else default

def _find_pdf_link(soup: BeautifulSoup):
    a = soup.find("a", attrs={"title": "Download PDF"})
    if a and a.get("href"):
        href = a["href"]
        return href if href.startswith("http") else f"https://openreview.net{href}"
    a2 = soup.find("a", href=re.compile(r"^/pdf\?id="))
    if a2 and a2.get("href"):
        return f"https://openreview.net{a2['href']}"
    return None

def _list_forums_from_group(group_id: str):
    url = f"https://openreview.net/group?id={group_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    forums = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/forum?id=" in href:
            forums.add(href if href.startswith("http") else f"https://openreview.net{href}")
    return sorted(forums)

def scrape_group(group_id: str):
    forums = _list_forums_from_group(group_id)
    print(f"Found {len(forums)} forums under {group_id}")
    for i, forum in enumerate(forums, 1):
        try:
            fr = requests.get(forum, timeout=60)
            fr.raise_for_status()
            soup = BeautifulSoup(fr.text, "html.parser")

            title = _extract_meta(soup, "citation_title") or soup.find("title").get_text(strip=True)
            authors = [m["content"].strip() for m in soup.find_all("meta", attrs={"name": "citation_author"}) if m.has_attr("content")]
            abstract = _extract_meta(soup, "citation_abstract") or ""
            date = _extract_meta(soup, "citation_publication_date") or _extract_meta(soup, "citation_online_date") or datetime.date.today().isoformat()
            pdf = _find_pdf_link(soup)
            if not pdf:
                print(f"[{i}/{len(forums)}] Skipping (no PDF): {title}")
                continue

            pr = requests.get(pdf, stream=True, timeout=120)
            pr.raise_for_status()
            key = _sanitize_filename(f"{title}.pdf")
            s3.upload_fileobj(pr.raw, BUCKET_NAME, key)
            _send_message(title, authors, date, abstract, BUCKET_NAME, key)
            print(f"[{i}/{len(forums)}] Uploaded + queued: {title}")
        except Exception as e:
            print(f"[{i}/{len(forums)}] Error on {forum}: {e}")

if __name__ == "__main__":
    group_id = os.getenv("OPENREVIEW_GROUP_ID", "ICLR.cc/2025/Conference")
    scrape_group(group_id)
