# fetch_arxiv.py
from __future__ import annotations

import json
import os
import uuid
import datetime
from datetime import date
from pathlib import Path

import arxiv
import boto3
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Environment / AWS clients
# -----------------------------------------------------------------------------
load_dotenv()  # reads .env in the current working directory

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = os.getenv("BUCKET_NAME")
QUEUE_URL = os.getenv("QUEUE_URL")  # optional

# --- New: configurable search ----
# Matches the arXiv UI search "LLM" in All fields, but broadens a bit to catch variants.
SEARCH_QUERY = os.getenv(
    "SEARCH_QUERY",
    '(LLM OR "large language model" OR "large language models") AND (cat:cs.CL OR cat:cs.LG OR cat:cs.AI OR cat:stat.ML)'
)
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))  # 50 like the UI "results per page"

if not BUCKET:
    raise RuntimeError("BUCKET_NAME is not set in .env")

# Let boto3 pick up AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from env automatically
s3 = boto3.client("s3", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION) if QUEUE_URL else None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def send_message(
    title: str,
    authors: list[str] | None,
    date_str: str,
    abstract: str,
    s3_bucket: str,
    s3_key: str,
    sqs_client,
    queue_url: str | None,
) -> None:
    """
    Send a JSON message to SQS (no-op if QUEUE_URL isn't set).
    Handles both Standard and FIFO queues (by checking .fifo suffix).
    """
    if not sqs_client or not queue_url:
        print("SQS not configured; skipping send_message.")
        return

    body = {
        "title": title,
        "authors": authors or [],
        "date": date_str,
        "abstract": abstract,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
    }

    params = {"QueueUrl": queue_url, "MessageBody": json.dumps(body)}
    if queue_url.endswith(".fifo"):
        params["MessageGroupId"] = "research-parsing"
        params["MessageDeduplicationId"] = str(uuid.uuid4())

    resp = sqs_client.send_message(**params)
    print("SQS MessageId:", resp.get("MessageId"))


def upload_code_to_s3(local_path: str, s3_bucket: str, s3_key: str | None = None) -> str:
    """
    Upload a local file (your code) to S3.
    Example s3_key: 'scripts/fetch_arxiv.py'
    """
    p = Path(local_path)
    if not s3_key:
        s3_key = f"scripts/{p.name}"

    s3.upload_file(
        Filename=str(p),
        Bucket=s3_bucket,
        Key=s3_key,
        ExtraArgs={"ContentType": "text/x-python", "ContentDisposition": "inline"},
    )
    print(f"Uploaded {p} to s3://{s3_bucket}/{s3_key}")
    return s3_key


def upload_json_to_s3(local_json: Path, s3_bucket: str, s3_key: str | None = None) -> str:
    if not s3_key:
        s3_key = f"arxiv/{local_json.name}"
    s3.upload_file(
        Filename=str(local_json),
        Bucket=s3_bucket,
        Key=s3_key,
        ExtraArgs={"ContentType": "application/json"},
    )
    print(f"Uploaded {local_json} to s3://{s3_bucket}/{s3_key}")
    return s3_key


# -----------------------------------------------------------------------------
# Main scrape -> save -> upload
# -----------------------------------------------------------------------------
def main() -> None:
    # Matches arXiv UI "Announcement date (newest first)"
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client(num_retries=5, delay_seconds=3)

    records: list[dict] = []
    for r in client.results(search):
        records.append(
            {
                "id": r.get_short_id(),
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "abstract": r.summary,
                "pdf_url": r.pdf_url,
                "arxiv_url": r.entry_id,
                "primary_category": r.primary_category,
                "published": r.published.isoformat() if r.published else None,
                "updated": r.updated.isoformat() if r.updated else None,
            }
        )

    # Save JSON locally
    out_path = Path(f"arxiv_{date.today()}_LLM.json")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} papers to {out_path}.")

    # Upload JSON to S3
    json_key = upload_json_to_s3(out_path, BUCKET)

    # Upload THIS code to S3 (same function name you asked to keep)
    code_key = upload_code_to_s3("fetch_arxiv.py", BUCKET, "scripts/fetch_arxiv.py")

    # Optionally notify via SQS that the JSON + code exist
    today = str(datetime.date.today())
    send_message(
        title="arxiv_dump_llm",
        authors=[],
        date_str=today,
        abstract=f"Saved {len(records)} arXiv records for query: {SEARCH_QUERY}",
        s3_bucket=BUCKET,
        s3_key=json_key,
        sqs_client=sqs,
        queue_url=QUEUE_URL,
    )
    send_message(
        title="code_upload",
        authors=[],
        date_str=today,
        abstract="Uploaded scraping code to S3.",
        s3_bucket=BUCKET,
        s3_key=code_key,
        sqs_client=sqs,
        queue_url=QUEUE_URL,
    )


if __name__ == "__main__":
    main()
