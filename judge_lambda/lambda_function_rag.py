"""
Enhanced Judge Lambda with RAG-based paper deduplication.
Uses vector embeddings and similarity search instead of exact matching.
"""

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
logger.info("Enhanced Judge Lambda with RAG started")

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))  # Default 85% similarity threshold

# AWS Clients
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# Initialize OpenSearch client with RAG capabilities
os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://","").replace("http://",""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)

def normalize_title(t: str) -> str:
    """Normalize title for consistent comparison."""
    return "".join(ch.lower() for ch in t if ch.isalnum() or ch.isspace()).strip().replace(" ", "_")

def sha256_text(t: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def ensure_vector_index():
    """Ensure OpenSearch index has proper vector mapping for RAG."""
    if not os_client.indices.exists(index=OPENSEARCH_INDEX):
        # Create index with vector mapping for RAG
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
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
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "date": {"type": "date"},
                    "s3_bucket": {"type": "keyword"},
                    "s3_key": {"type": "keyword"},
                    "sha_abstract": {"type": "keyword"},
                    "decision": {"type": "keyword"},
                    "reason": {"type": "text"},
                    "relevance": {"type": "keyword"},
                    "novelty": {"type": "keyword"},
                    "similarity_score": {"type": "float"},
                    "most_similar_paper": {"type": "keyword"},
                    "ingested_at": {"type": "date"}
                }
            }
        }
        os_client.indices.create(index=OPENSEARCH_INDEX, body=body)
        logger.info("Created vector-enabled index for RAG")

def generate_embedding(text: str) -> List[float]:
    """Generate vector embedding using Amazon Titan Embeddings."""
    try:
        body = json.dumps({
            "inputText": text
        })
        
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=body,
            contentType="application/json"
        )
        
        result = json.loads(response["body"].read())
        return result["embedding"]
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

def search_similar_papers_rag(abstract: str, size: int = 5) -> List[Dict]:
    """Search for similar papers using vector similarity (RAG approach)."""
    try:
        # Generate embedding for the abstract
        embedding = generate_embedding(abstract)
        if not embedding:
            logger.warning("Failed to generate embedding, falling back to text search")
            return []
        
        # Perform k-NN search
        query = {
            "knn": {
                "field": "abstract_embedding",
                "query_vector": embedding,
                "k": size,
                "num_candidates": 100
            }
        }
        
        response = os_client.search(
            index=OPENSEARCH_INDEX,
            body={"query": query, "size": size}
        )
        
        similar_papers = []
        for hit in response['hits']['hits']:
            paper = hit['_source']
            paper['_id'] = hit['_id']
            paper['similarity_score'] = hit['_score']
            similar_papers.append(paper)
        
        logger.info(f"Found {len(similar_papers)} similar papers via RAG")
        return similar_papers
        
    except Exception as e:
        logger.error(f"Error in RAG similarity search: {e}")
        return []

def is_paper_redundant_rag(title: str, abstract: str) -> Dict[str, any]:
    """Check if paper is redundant using RAG-based similarity search."""
    try:
        # Search for similar papers using vector embeddings
        similar_papers = search_similar_papers_rag(abstract, size=10)
        
        if not similar_papers:
            return {
                "is_redundant": False,
                "reason": "No similar papers found",
                "max_similarity": 0.0,
                "similar_papers": []
            }
        
        # Get the most similar paper
        most_similar = similar_papers[0]
        max_similarity = most_similar.get('similarity_score', 0.0)
        
        # Check if similarity exceeds threshold
        is_redundant = max_similarity >= SIMILARITY_THRESHOLD
        
        return {
            "is_redundant": is_redundant,
            "reason": f"Most similar paper has {max_similarity:.3f} similarity (threshold: {SIMILARITY_THRESHOLD})",
            "max_similarity": max_similarity,
            "most_similar_paper": most_similar.get('title', 'Unknown'),
            "similar_papers": similar_papers[:3]  # Top 3 similar papers
        }
        
    except Exception as e:
        logger.error(f"Error checking redundancy with RAG: {e}")
        return {
            "is_redundant": False,
            "reason": f"Error during RAG redundancy check: {str(e)}",
            "max_similarity": 0.0,
            "similar_papers": []
        }

def evaluate_paper_with_claude(title: str, abstract: str) -> Dict[str, str]:
    """Evaluate paper relevance and novelty using Claude via Bedrock."""
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

def index_paper_with_embedding(paper_data: Dict) -> bool:
    """Index paper with its vector embedding."""
    try:
        abstract = paper_data.get('abstract', '')
        if not abstract:
            logger.warning("No abstract provided, cannot generate embedding")
            return False
        
        # Generate embedding
        embedding = generate_embedding(abstract)
        if not embedding:
            logger.error("Failed to generate embedding for paper")
            return False
        
        # Add embedding to paper data
        paper_data['abstract_embedding'] = embedding
        
        # Index the paper
        paper_id = paper_data.get('title_normalized', '')
        os_client.index(
            index=OPENSEARCH_INDEX,
            id=paper_id,
            body=paper_data,
            refresh=True
        )
        
        logger.info(f"Successfully indexed paper with embedding: {paper_data.get('title', 'Unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Error indexing paper with embedding: {e}")
        return False

def lambda_handler(event, context):
    """Enhanced Lambda handler with RAG-based redundancy detection."""
    ensure_vector_index()
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
            
            # Check for redundancy using RAG
            redundancy_check = is_paper_redundant_rag(title, abstract)
            
            if redundancy_check["is_redundant"]:
                logger.info(f"Paper rejected as redundant | {title} | {redundancy_check['reason']}")
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
                    "similarity_score": redundancy_check["max_similarity"],
                    "most_similar_paper": redundancy_check.get("most_similar_paper", ""),
                    "ingested_at": int(time.time() * 1000)
                }
                
                # Index with embedding
                if index_paper_with_embedding(doc):
                    logger.info(f"Paper accepted and indexed with RAG | {title}")
                else:
                    logger.error(f"Failed to index paper | {title}")
            else:
                logger.info(f"Paper rejected (irrelevant or not novel) | {title} | {reason}")

        except Exception as e:
            logger.exception(f"Failed to process message {msg_id}: {e}")
            failures.append({"itemIdentifier": msg_id})

    logger.info(f"Enhanced Judge Lambda complete. Failures: {len(failures)}")
    return {"batchItemFailures": failures}
