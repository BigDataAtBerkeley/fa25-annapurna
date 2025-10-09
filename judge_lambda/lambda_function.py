import boto3
import json
import os
import time
import hashlib
import logging
from typing import List, Dict
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# ===== Logging =====
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info("✅ Judge Lambda started")

# ===== Environment =====
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers")

# ===== AWS Clients =====
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://","").replace("http://",""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)
# ===== Helper functions =====

## normalize title so that we can easily check OpenSearch every time we scrape a paper for if it alrdy exists
def normalize_title(t: str) -> str:
    """
    Normalize a paper's title into a consistent, comparable format for OpenSearch deduplication.
    Example:
        "A Study on LLM Scaling Laws!" → "a_study_on_llm_scaling_laws"
    """
    return "".join(ch.lower() for ch in t if ch.isalnum() or ch.isspace()).strip().replace(" ", "_")


## this will take the abstract section of the text and convert it to a hash version of the str (for efficient storage)
def sha256_text(t: str) -> str:
    """
    Compute a SHA-256 hash of a text string.
    Args:
        t: The text string (e.g., a paper abstract).
    Returns:
        Hexadecimal digest of the SHA-256 hash (a 64-character string).
    Example:
        sha256_text("hello") → "2cf24dba5fb0..."
    """
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def ensure_index():
    """
    Ensure that the OpenSearch index for papers exists. If not, create it with the proper schema.
    """
    if not os_client.indices.exists(index=OPENSEARCH_INDEX):
        body = {
            "settings": {"index": {"number_of_shards": 1}},
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "title_normalized": {"type": "keyword"},
                    "authors": {"type": "keyword"},
                    "abstract": {"type": "text"},
                    "date": {"type": "date"},
                    "s3_bucket": {"type": "keyword"},
                    "s3_key": {"type": "keyword"},
                    "sha_abstract": {"type": "keyword"},
                    "decision": {"type": "keyword"},
                    "reason": {"type": "text"},
                    "relevance": {"type": "keyword"},
                    "novelty": {"type": "keyword"},
                    "ingested_at": {"type": "date"}
                }
            }
        }
        os_client.indices.create(index=OPENSEARCH_INDEX, body=body)
        logger.info("✅ Created index")


def is_duplicate(title_norm: str, sha_abs: str) -> bool:
    """
    Check if a paper already exists in OpenSearch based on title or abstract hash.

    Purpose:
        - Avoid storing duplicate papers (same title or abstract).
        - Uses both title and abstract hash for redundancy (title alone may differ slightly).
    Args:
        title_norm: Normalized version of the paper title.
        sha_abs: SHA-256 hash of the paper's abstract.
    Returns:
        Boolean → True if duplicate found, else False.
    """
    q = {
        "query": {
            "bool": {
                "should": [ ## bascially just an "OR" condition (if titles are the same OR abstracts the same)
                    {"term": {"title_normalized": title_norm}},
                    {"term": {"sha_abstract": sha_abs}}
                ],
                "minimum_should_match": 1
            }
        }
    }
    try:
        res = os_client.search(index=OPENSEARCH_INDEX, body=q)
        return res["hits"]["total"]["value"] > 0
    except Exception:
        # If OpenSearch fails (e.g., timeout, index missing, etc.), treat as non-duplicate to keep pipeline running.
        return False


# ===== Claude via Bedrock =====
def evaluate_paper_with_claude(title: str, abstract: str) -> Dict[str, str]:
    """
    Ask Claude (via AWS Bedrock) to evaluate the paper for relevance and novelty.

    Purpose:
        - Automate the "judge" step: determine whether the paper is worth indexing further.
        - Helps prioritize papers that are relevant to LLM/AI/ML research and not repetitive.

    Expected output JSON format:
        {
            "relevance": "high" | "medium" | "low" | "unknown",
            "novelty": "high" | "medium" | "low" | "unknown",
            "reason": "Short explanation of why"
        }

    Model parameters:
        - modelId: "anthropic.claude-3-haiku-20240307" (fast + cost-efficient).
        - maxTokens: 200 (enough for a concise JSON).
        - temperature: 0.3 (low randomness → consistent and focused answers).

    Args:
        title: Paper title.
        abstract: Paper abstract text.

    Returns:
        Dictionary containing "relevance", "novelty", and "reason".
    """
    prompt = f"""
You are an expert ML researcher. 
Read the paper below and decide:
1. Is it relevant to current LLM, AI, or ML research?
2. Is it novel (introduces new techniques, datasets, or methods)?

Paper:
Title: {title}
Abstract: {abstract}

Answer in JSON with fields: relevance, novelty, reason.
Keep it short.
"""
    try:
        # Bedrock request payload
        body = json.dumps({
            "modelId": "anthropic.claude-3-haiku-20240307",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ],
            "maxTokens": 200,
            "temperature": 0.3
        })

        # Invoke Claude via Bedrock client
        response = bedrock.invoke_model(body=body)

        # Bedrock responses are streamed; extract body text safely
        text = json.loads(response["body"].read())["content"][0]["text"].strip()
        
        # Claude is instructed to output JSON; parse it into a Python dict
        return json.loads(text)

    except Exception as e:
        logger.warning(f"Claude via Bedrock failed: {e}")
        return {"relevance": "unknown", "novelty": "unknown", "reason": "Claude evaluation failed"}


# ===== Lambda handler =====
def lambda_handler(event, context):
    ensure_index()
    failures = []

    for record in event.get("Records", []):
        msg_id = record["messageId"]

        try:
            body = json.loads(record["body"])
            title = body["title"]
            abstract = body.get("abstract", "")
            authors = body.get("authors", [])
            s3_bucket = body["s3_bucket"]
            s3_key = body["s3_key"]
            date = body.get("date")

            title_norm = normalize_title(title)
            sha_abs = sha256_text(abstract)
            if is_duplicate(title_norm, sha_abs):
                logger.info(f"⚠️ Duplicate skipped | {title}")
                continue

            # Evaluate with Claude (Bedrock)
            evaluation = evaluate_paper_with_claude(title, abstract)
            relevance = evaluation.get("relevance", "unknown").lower()
            novelty = evaluation.get("novelty", "unknown").lower()
            reason = evaluation.get("reason", "No reason provided.")

            # Only store relevant + novel papers
            if relevance == "yes" and novelty == "yes":
                doc = {
                    "title": title,
                    "title_normalized": title_norm,
                    "authors": authors,
                    "abstract": abstract,
                    "date": date.replace("/", "-") if date else None,
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "sha_abstract": sha_abs,
                    "decision": "accept",
                    "reason": reason,
                    "relevance": relevance,
                    "novelty": novelty,
                    "ingested_at": int(time.time() * 1000)
                }
                os_client.index(index=OPENSEARCH_INDEX, body=doc, refresh=True)
                logger.info(f"✅ Indexed | {title}")
            else:
                logger.info(f"🛑 Skipped (irrelevant or not novel) | {title} | {reason}")

        except Exception as e:
            logger.exception(f"❌ Failed to process message {msg_id}: {e}")
            failures.append({"itemIdentifier": msg_id})

    logger.info(f"✅ Lambda complete. Failures: {len(failures)}")
    return {"batchItemFailures": failures}
