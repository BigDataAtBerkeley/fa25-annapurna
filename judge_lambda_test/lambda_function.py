# This is a simlation version of the judge lambda that does not modify any data in OpenSearch.
# Requests are sent using setup_test.py and require existing documents in OpenSearch.

import boto3
import json
import os
import time
import hashlib
import logging
from typing import List, Dict
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# === Logging setup ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info("Tester Lambda (Simulation) started")

# === Environment variables ===
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

# === AWS Clients ===
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", ""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)

# === Utility Functions ===
def normalize_title(t: str) -> str:
    return "".join(ch.lower() for ch in t if ch.isalnum() or ch.isspace()).strip().replace(" ", "_")

def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def generate_embedding(text: str) -> List[float]:
    """Generate embedding via Titan Embed (Bedrock)."""
    try:
        body = json.dumps({"inputText": text})
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=body,
            contentType="application/json"
        )
        result = json.loads(response["body"].read())
        return result.get("embedding", [])
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return []

def search_similar_papers_rag(abstract: str, size: int = 10) -> List[Dict]:
    """Perform KNN search in OpenSearch for similar papers."""
    try:
        embedding = generate_embedding(abstract)
        if not embedding:
            return []
        query = {
            "knn": {
                "field": "abstract_embedding",
                "query_vector": embedding,
                "k": size,
                "num_candidates": 100
            }
        }
        res = os_client.search(index=OPENSEARCH_INDEX, body={"query": query, "size": size})
        hits = res.get("hits", {}).get("hits", [])
        out = []
        for hit in hits:
            src = hit.get("_source", {})
            src["_id"] = hit.get("_id")
            src["similarity_score"] = hit.get("_score")
            out.append(src)
        return out
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        return []

def evaluate_paper_with_claude(title: str, abstract: str) -> Dict[str, str]:
    """Ask Claude to assess relevance and novelty."""
    prompt = f"""
You are an expert ML researcher.
Read the paper below and decide:
1. Is it relevant to current LLM, AI, or ML research?
2. Is it novel (introduces new techniques for training/inference)?
Exclude surveys or benchmark proposals.

Paper:
Title: {title}
Abstract: {abstract}

Respond in JSON: {{ "relevance": "yes/no", "novelty": "yes/no", "reason": "..." }}
"""
    try:
        body = json.dumps({
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": 200,
            "temperature": 0.3
        })
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=body
        )
        text = json.loads(response["body"].read())["content"][0]["text"].strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Claude evaluation failed: {e}")
        return {"relevance": "unknown", "novelty": "unknown", "reason": "Claude evaluation failed"}

def simulate_paper_decision(title: str, abstract: str) -> Dict[str, any]:
    """Simulate full judge pipeline without modifying data."""
    title_norm = normalize_title(title)
    sha_abs = sha256_text(abstract)

    # === Step 1: Similarity Check ===
    similar_papers = search_similar_papers_rag(abstract, size=10)
    if not similar_papers or len(similar_papers) < 2:
        sim_score = 0.0
        sim_reason = "No comparable papers found."
    else:
        # Skip first (self-match) and use second-most similar
        second_best = similar_papers[1]
        sim_score = second_best.get("similarity_score", 0.0) or 0.0
        sim_reason = f"Second-most similar: '{second_best.get('title', 'Unknown')}' ({sim_score:.3f} similarity)"

    is_redundant = sim_score >= SIMILARITY_THRESHOLD
    if is_redundant:
        return {
            "title": title,
            "decision": "reject",
            "rejected_by": "rag",
            "reason": f"Similarity {sim_score:.3f} ≥ threshold {SIMILARITY_THRESHOLD}. {sim_reason}"
        }

    # === Step 2: Claude Evaluation ===
    eval_result = evaluate_paper_with_claude(title, abstract)
    rel = eval_result.get("relevance", "unknown").lower()
    nov = eval_result.get("novelty", "unknown").lower()
    reason = eval_result.get("reason", "No reason provided")

    if rel == "yes" and nov == "yes":
        return {
            "title": title,
            "decision": "accept",
            "rejected_by": None,
            "reason": f"Accepted — {reason}",
            "similarity": sim_score,
            "similar_paper": sim_reason
        }
    else:
        return {
            "title": title,
            "decision": "reject",
            "rejected_by": "claude",
            "reason": f"Rejected by Claude — {reason}",
            "similarity": sim_score,
            "similar_paper": sim_reason
        }

# === Lambda Handler (Simulation Mode) ===
def lambda_handler(event, context):
    logger.info("Running tester simulation Lambda...")
    results = []

    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            title = body.get("title", "Untitled")
            abstract = body.get("abstract", "")
            authors = body.get("authors", [])
            s3_bucket = body.get("s3_bucket")
            s3_key = body.get("s3_key")

            result = simulate_paper_decision(title, abstract)
            result.update({
                "authors": authors,
                "s3_bucket": s3_bucket,
                "s3_key": s3_key
            })
            results.append(result)

            logger.info(f"Simulated decision for '{title}': {result['decision']} ({result['reason']})")

        except Exception as e:
            logger.exception(f"Error processing record: {e}")
            results.append({"error": str(e)})

    logger.info("Simulation complete.")
    return {"simulation_results": results}