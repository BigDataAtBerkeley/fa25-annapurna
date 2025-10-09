import boto3
import json
import os
import time
import hashlib
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info("Judge Lambda started")

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers")

# AWS Clients
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

#Transformer model for embeddings
transformer = SentenceTransformer("all-MiniLM-L6-v2")

os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://","").replace("http://",""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)

def normalize_title(t: str) -> str:
    return "".join(ch.lower() for ch in t if ch.isalnum() or ch.isspace()).strip().replace(" ", "_")

def get_embedding(model, text: str):
    return model.encode(text, normalize_embeddings=True).tolist()

def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def ensure_index():
    if not os_client.indices.exists(index=OPENSEARCH_INDEX):
        body = {
            "settings": {"index": {"number_of_shards": 1}},
            "mappings": {"properties": {
                "title": {"type": "text"},
                "title_normalized": {"type": "keyword"},
                "authors": {"type": "keyword"},
                "abstract": {"type": "text"},
                "date": {"type": "date"},
                "s3_bucket": {"type": "keyword"},
                "s3_key": {"type": "keyword"},
                "sha_abstract": {"type": "keyword"},
                "embedding" : {"type": "knn_vector", "dimension": 384},
                "decision": {"type": "keyword"},
                "reason": {"type": "text"},
                "relevance": {"type": "keyword"},
                "novelty": {"type": "keyword"},
                "ingested_at": {"type": "date"}
            }}
        }
        os_client.indices.create(index=OPENSEARCH_INDEX, body=body)
        logger.info("Created index")
        

# Redundancy check - filters out duplicates and papers that are very similar to existing ones.
def is_duplicate(title_norm: str, abstract: str, transformer, os_client, OPENSEARCH_INDEX: str) -> bool:
    try:
        exact_query = {"query": {"bool": {"should": [
                    {"term": {"title_normalized": title_norm}},
                    {"term": {"sha_abstract": sha256_text(abstract)}}],
                    "minimum_should_match": 1}}}

        res = os_client.search(index=OPENSEARCH_INDEX, body=exact_query)
        if res["hits"]["total"]["value"] > 0:
            return True  

        embedding = get_embedding(transformer, f"{title_norm} {abstract}")

        knn_query = {"size": 3, "query": {"knn": {"embedding": {"vector": embedding,"k": 3}}}}

        res = os_client.search(index=OPENSEARCH_INDEX, body=knn_query)
        hits = res["hits"]["hits"]

        similarity_threshold = 0.9 
        return any(hit["_score"] >= similarity_threshold for hit in hits)

    except Exception as e:
        return False

# Claude via Bedrock evaluation
def evaluate_paper_with_claude(title: str, abstract: str) -> Dict[str, str]:
    """Ask Claude (via Bedrock) to assess relevance and novelty."""
    prompt = f"""
You are an expert ML researcher. 
Read the paper below and decide:
1. Is it relevant to current LLM, AI, or ML research?
2. Is it novel (introduces new techniques or methods, specifically for training or inference)?
Exclude papers that are surveys, summaries, or non-technical, such as ethics or proposals for new benchmarks. 

Paper:
Title: {title}
Abstract: {abstract}

Answer in JSON with fields: relevance, novelty, reason.
Keep answers to 'yes' or 'no' for relevance and novelty.
Keep it short.
"""
    try:
        body = json.dumps({
            "modelId": "anthropic.claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "maxTokens": 200,
            "temperature": 0.3
        })
        response = bedrock.invoke_model(body=body)
        text = json.loads(response["body"].read())["content"][0]["text"].strip()
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
                logger.info(f"Duplicate skipped | {title}")
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
                logger.info(f"Indexed | {title}")
            else:
                logger.info(f"Skipped (irrelevant or not novel) | {title} | {reason}")

        except Exception as e:
            logger.exception(f"Failed to process message {msg_id}: {e}")
            failures.append({"itemIdentifier": msg_id})

    logger.info(f"Lambda complete. Failures: {len(failures)}")
    return {"batchItemFailures": failures}
