import boto3
import json
import os
import time
import hashlib
import logging
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
CODE_EVAL_QUEUE_URL = os.getenv("CODE_EVAL_QUEUE_URL")  # SQS queue for code generation
DISCARDED_BUCKET = os.getenv("DISCARDED_BUCKET", "discarded-papers")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

# AWS Clients
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
sqs_client = boto3.client("sqs", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)

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

def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def ensure_index():
    if not os_client.indices.exists(index=OPENSEARCH_INDEX):
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {"properties": {
                "title": {"type": "text"},
                "title_normalized": {"type": "keyword"},
                "authors": {"type": "keyword"},
                "abstract": {"type": "text"},
                "abstract_embedding": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 128, "m": 24}
                    }
                },
                "date": {"type": "date"},
                "s3_bucket": {"type": "keyword"},
                "s3_key": {"type": "keyword"},
                "sha_abstract": {"type": "keyword"},
                "decision": {"type": "keyword"},
                "rejected_by": {"type": "keyword"},
                "reason": {"type": "text"},
                "relevance": {"type": "keyword"},
                "novelty": {"type": "keyword"},
                "ingested_at": {"type": "date"}
            }}
        }
        os_client.indices.create(index=OPENSEARCH_INDEX, body=body)
        logger.info("Created vector-enabled index")
    else:
        # Best-effort ensure required fields exist
        try:
            mapping = os_client.indices.get_mapping(index=OPENSEARCH_INDEX)
            props = mapping.get(OPENSEARCH_INDEX, {}).get("mappings", {}).get("properties", {})
            missing_updates = {}
            if "abstract_embedding" not in props:
                missing_updates.setdefault("properties", {})["abstract_embedding"] = {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 128, "m": 24}
                    }
                }
            if "rejected_by" not in props:
                missing_updates.setdefault("properties", {})["rejected_by"] = {"type": "keyword"}
            if missing_updates:
                os_client.indices.put_mapping(index=OPENSEARCH_INDEX, body=missing_updates)
                logger.info("Updated index mapping with missing fields")
        except Exception as e:
            logger.warning(f"Failed to ensure index mapping: {e}")

def generate_embedding(text: str) -> List[float]:
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

def search_similar_papers_rag(abstract: str, size: int = 5) -> List[Dict]:
    try:
        embedding = generate_embedding(abstract)
        if not embedding:
            return []
        try:
            embedding = [float(x) for x in embedding]
        except Exception:
            pass

        try:
            mapping = os_client.indices.get_mapping(index=OPENSEARCH_INDEX)
            props = mapping.get(OPENSEARCH_INDEX, {}).get("mappings", {}).get("properties", {})
            dim = int(props.get("abstract_embedding", {}).get("dimension", len(embedding)))
        except Exception:
            dim = len(embedding)
        if len(embedding) > dim:
            embedding = embedding[:dim]
        elif len(embedding) < dim:
            embedding = embedding + [0.0] * (dim - len(embedding))

        query_primary = {
            "knn": {
                "abstract_embedding": {
                    "vector": embedding,
                    "k": size
                }
            }
        }
        try:
            res = os_client.search(index=OPENSEARCH_INDEX, body={"query": query_primary, "size": size})
        except Exception as e1:
            logger.warning(f"Primary kNN query failed, retrying with field/query_vector: {e1}")
            query_fallback = {
                "knn": {
                    "field": "abstract_embedding",
                    "query_vector": embedding,
                    "k": size
                }
            }
            res = os_client.search(index=OPENSEARCH_INDEX, body={"query": query_fallback, "size": size})
        out = []
        for hit in res.get('hits', {}).get('hits', []):
            src = hit.get('_source', {})
            src['_id'] = hit.get('_id')
            src['similarity_score'] = hit.get('_score')
            out.append(src)
        return out
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        return []

def is_paper_redundant_rag(title: str, abstract: str) -> Dict[str, any]:
    similar = search_similar_papers_rag(abstract, size=10)
    if not similar:
        return {"is_redundant": False, "reason": "No similar papers found", "max_similarity": 0.0}
    most_similar = similar[0]
    max_sim = most_similar.get('similarity_score', 0.0) or 0.0
    return {
        "is_redundant": max_sim >= SIMILARITY_THRESHOLD,
        "reason": f"Most similar has {max_sim:.3f} similarity (threshold={SIMILARITY_THRESHOLD})",
        "max_similarity": max_sim,
        "most_similar_paper": most_similar.get('title', 'Unknown')
    }

def index_paper_document(doc_id: str, doc: Dict) -> str:
    try:
        result = os_client.index(index=OPENSEARCH_INDEX, id=doc_id, body=doc, refresh=True)
        return result.get('_id', doc_id)
    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        return doc_id

def write_discard_record(doc_id: str, rejected_by: str, reason: str, original: Dict, msg_id: str) -> None:
    try:
        now = datetime.utcnow()
        key = f"rejected/{doc_id}.json"
        try:
            s3_client.head_object(Bucket=DISCARDED_BUCKET, Key=key)
            return
        except s3_client.exceptions.ClientError:
            pass
        record = {
            "decision": "reject",
            "rejected_by": rejected_by,
            "reason": reason,
            "message_id": msg_id,
            "evaluated_at": now.isoformat(),
            "paper": original
        }
        s3_client.put_object(Bucket=DISCARDED_BUCKET, Key=key, Body=json.dumps(record).encode('utf-8'), ContentType='application/json')
        logger.info(f"Wrote discard record to s3://{DISCARDED_BUCKET}/{key}")
    except Exception as e:
        logger.error(f"Failed to write discard record: {e}")
        

# Redundancy check - filters out duplicates and papers that are very similar to existing ones.
def is_duplicate(title_norm: str, sha_abs: str) -> bool:
    try:
        exact_query = {"query": {"bool": {"should": [
                    {"term": {"title_normalized": title_norm}},
                    {"term": {"sha_abstract": sha_abs}}],
                    "minimum_should_match": 1}}}

        res = os_client.search(index=OPENSEARCH_INDEX, body=exact_query)
        return res["hits"]["total"]["value"] > 0
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
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": 200,
            "temperature": 0.3
        })
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            # modelId="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )
        text = json.loads(response["body"].read())["content"][0]["text"].strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Claude via Bedrock failed: {e}")
        return {"relevance": "unknown", "novelty": "unknown", "reason": "Claude evaluation failed"}

def send_to_code_eval_queue(paper_id: str, paper_data: Dict) -> bool:
    """
    Send paper to code-evaluation queue (for code generation when 10 accumulate).
    
    Args:
        paper_id: OpenSearch document ID
        paper_data: Paper metadata
        
    Returns:
        True if successful, False otherwise
    """
    if not CODE_EVAL_QUEUE_URL:
        logger.warning("CODE_EVAL_QUEUE_URL not set, skipping queue")
        return False
    
    try:
        message = {
            "paper_id": paper_id,
            "action": "generate_by_id",
            "paper_title": paper_data.get("title"),
            "queued_at": datetime.now().isoformat()
        }
        
        sqs_client.send_message(
            QueueUrl=CODE_EVAL_QUEUE_URL,
            MessageBody=json.dumps(message),
            MessageGroupId=paper_id,
            MessageDeduplicationId=f"{paper_id}-{int(time.time() * 1000)}"
        )
        
        logger.info(f"Sent to code-eval queue: {paper_data.get('title')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send to code-eval queue: {e}")
        return False

# Lambda Handler
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
            # Precompute embedding for indexing (used for RAG storage and future searches)
            precomputed_embedding = generate_embedding(abstract)
            doc_id = sha256_text(f"{title_norm}:{sha_abs}")
            
            # Exact duplicate pre-check
            if is_duplicate(title_norm, sha_abs):
                reason = "Exact duplicate by title_normalized or sha_abstract"
                write_discard_record(doc_id, "exact_duplicate", reason, body, msg_id)
                reject_doc = {
                    "title": title,
                    "title_normalized": title_norm,
                    "authors": authors,
                    "abstract": abstract,
                    "date": date.replace("/", "-") if date else None,
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "sha_abstract": sha_abs,
                    "decision": "reject",
                    "rejected_by": "exact_duplicate",
                    "reason": reason,
                    "relevance": "unknown",
                    "novelty": "unknown",
                    "ingested_at": int(time.time() * 1000)
                }
                if precomputed_embedding:
                    reject_doc["abstract_embedding"] = precomputed_embedding
                index_paper_document(doc_id, reject_doc)
                logger.info(f"Skipped (exact duplicate) | {title} | {reason}")
                continue

            # RAG redundancy pre-check
            redundancy = is_paper_redundant_rag(title, abstract)
            if redundancy.get("is_redundant"):
                reason = redundancy.get("reason", "RAG redundancy")
                write_discard_record(doc_id, "rag", reason, body, msg_id)
                reject_doc = {
                    "title": title,
                    "title_normalized": title_norm,
                    "authors": authors,
                    "abstract": abstract,
                    "date": date.replace("/", "-") if date else None,
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "sha_abstract": sha_abs,
                    "decision": "reject",
                    "rejected_by": "rag",
                    "reason": reason,
                    "relevance": "unknown",
                    "novelty": "unknown",
                    "ingested_at": int(time.time() * 1000)
                }
                if precomputed_embedding:
                    reject_doc["abstract_embedding"] = precomputed_embedding
                index_paper_document(doc_id, reject_doc)
                logger.info(f"Rejected by RAG | {title} | {reason}")
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
                # Add embedding for accepted docs
                if precomputed_embedding:
                    doc["abstract_embedding"] = precomputed_embedding
                # Index in OpenSearch
                paper_id = index_paper_document(doc_id, doc)
                logger.info(f"Indexed | {title} | ID: {paper_id}")
                
                # Send to code-evaluation queue (will trigger code gen when 10 accumulate)
                send_to_code_eval_queue(paper_id, doc)
                
            else:
                write_discard_record(doc_id, "claude", reason, body, msg_id)
                reject_doc = {
                    "title": title,
                    "title_normalized": title_norm,
                    "authors": authors,
                    "abstract": abstract,
                    "date": date.replace("/", "-") if date else None,
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "sha_abstract": sha_abs,
                    "decision": "reject",
                    "rejected_by": "claude",
                    "reason": reason,
                    "relevance": relevance,
                    "novelty": novelty,
                    "ingested_at": int(time.time() * 1000)
                }
                if precomputed_embedding:
                    reject_doc["abstract_embedding"] = precomputed_embedding
                index_paper_document(doc_id, reject_doc)
                logger.info(f"Skipped (irrelevant or not novel) | {title} | {reason}")

        except Exception as e:
            logger.exception(f"Failed to process message {msg_id}: {e}")
            failures.append({"itemIdentifier": msg_id})

    logger.info(f"Lambda complete. Failures: {len(failures)}")
    return {"batchItemFailures": failures}
