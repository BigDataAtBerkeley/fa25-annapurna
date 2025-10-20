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
logger.info("RAG-Enhanced Judge Lambda started")

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))  # 95% similarity for redundancy
EMBEDDINGS_MODEL_ID = os.getenv("EMBEDDINGS_MODEL_ID", "amazon.titan-embed-text-v1")

# AWS Clients
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
            }}
        }
        os_client.indices.create(index=OPENSEARCH_INDEX, body=body)
        logger.info("Created index with RAG vector support")
    else:
        # Check if the index has the vector mapping
        try:
            mapping = os_client.indices.get_mapping(index=OPENSEARCH_INDEX)
            properties = mapping[OPENSEARCH_INDEX]['mappings']['properties']
            
            if 'abstract_embedding' not in properties:
                logger.warning("Index exists but doesn't have vector mapping. RAG functionality may not work properly.")
                logger.info("Consider recreating the index or adding the vector mapping manually.")
            else:
                logger.info("Index exists with RAG vector support")
        except Exception as e:
            logger.warning(f"Could not check index mapping: {e}")
        

# Generate vector embedding using Amazon Titan Embeddings
def generate_embedding(text: str) -> List[float]:
    """Generate vector embedding using Amazon Titan Embeddings."""
    try:
        body = json.dumps({
            "inputText": text
        })
        
        response = bedrock.invoke_model(
            modelId=EMBEDDINGS_MODEL_ID,
            body=body,
            contentType="application/json"
        )
        
        result = json.loads(response["body"].read())
        return result["embedding"]
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return []

# RAG-based redundancy check using vector similarity
def is_paper_redundant_rag(title: str, abstract: str) -> Dict[str, any]:
    """Check if paper is redundant using RAG-based similarity search."""
    try:
        # Generate embedding for the abstract
        embedding = generate_embedding(abstract)
        if not embedding:
            logger.warning("Failed to generate embedding, falling back to exact matching")
            return {"is_redundant": False, "reason": "Embedding generation failed", "max_similarity": 0.0}
        
        # Perform k-NN search for similar papers
        query = {
            "knn": {
                "field": "abstract_embedding",
                "query_vector": embedding,
                "k": 10,
                "num_candidates": 100
            }
        }
        
        response = os_client.search(
            index=OPENSEARCH_INDEX,
            body={"query": query, "size": 10}
        )
        
        similar_papers = []
        for hit in response['hits']['hits']:
            paper = hit['_source']
            paper['_id'] = hit['_id']
            paper['similarity_score'] = hit['_score']
            similar_papers.append(paper)
        
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

# Add embeddings to existing papers that don't have them
def add_embeddings_to_existing_papers():
    """Add embeddings to existing papers that don't have them yet."""
    try:
        # Find papers without embeddings
        query = {
            "query": {
                "bool": {
                    "must_not": {
                        "exists": {"field": "abstract_embedding"}
                    }
                }
            },
            "size": 100
        }
        
        response = os_client.search(index=OPENSEARCH_INDEX, body=query)
        papers_without_embeddings = response['hits']['hits']
        
        if not papers_without_embeddings:
            logger.info("All existing papers already have embeddings")
            return
        
        logger.info(f"Found {len(papers_without_embeddings)} papers without embeddings, adding them...")
        
        for hit in papers_without_embeddings:
            paper_id = hit['_id']
            paper_data = hit['_source']
            abstract = paper_data.get('abstract', '')
            
            if not abstract:
                logger.warning(f"Paper {paper_id} has no abstract, skipping embedding generation")
                continue
            
            # Generate embedding
            embedding = generate_embedding(abstract)
            if not embedding:
                logger.warning(f"Failed to generate embedding for paper {paper_id}")
                continue
            
            # Update the paper with embedding
            os_client.update(
                index=OPENSEARCH_INDEX,
                id=paper_id,
                body={
                    "doc": {
                        "abstract_embedding": embedding
                    }
                }
            )
            logger.info(f"Added embedding to paper: {paper_data.get('title', 'Unknown')}")
        
        logger.info("Finished adding embeddings to existing papers")
        
    except Exception as e:
        logger.error(f"Error adding embeddings to existing papers: {e}")

# Batch process embeddings for better performance
def batch_generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts efficiently."""
    embeddings = []
    
    for text in texts:
        embedding = generate_embedding(text)
        embeddings.append(embedding)
    
    return embeddings

# Enhanced RAG search with better error handling
def search_similar_papers_enhanced(abstract: str, size: int = 10) -> List[Dict]:
    """Enhanced search for similar papers with better error handling."""
    try:
        # Generate embedding for the abstract
        embedding = generate_embedding(abstract)
        if not embedding:
            logger.warning("Failed to generate embedding, falling back to text search")
            return []
        
        # Perform k-NN search with enhanced parameters
        query = {
            "knn": {
                "field": "abstract_embedding",
                "query_vector": embedding,
                "k": size,
                "num_candidates": min(size * 10, 200)  # Dynamic candidate count
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
        logger.error(f"Error in enhanced RAG similarity search: {e}")
        return []

# RAG-based paper similarity analysis
def analyze_paper_similarity(title: str, abstract: str) -> Dict[str, any]:
    """Comprehensive similarity analysis using RAG."""
    try:
        # Search for similar papers
        similar_papers = search_similar_papers_enhanced(abstract, size=15)
        
        if not similar_papers:
            return {
                "is_redundant": False,
                "reason": "No similar papers found",
                "max_similarity": 0.0,
                "similarity_distribution": {},
                "top_similar_papers": []
            }
        
        # Analyze similarity distribution
        similarity_scores = [paper['similarity_score'] for paper in similar_papers]
        max_similarity = max(similarity_scores)
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Categorize similarity levels
        high_similarity = len([s for s in similarity_scores if s >= 0.95])
        medium_similarity = len([s for s in similarity_scores if 0.85 <= s < 0.95])
        low_similarity = len([s for s in similarity_scores if s < 0.85])
        
        similarity_distribution = {
            "high_similarity_count": high_similarity,
            "medium_similarity_count": medium_similarity,
            "low_similarity_count": low_similarity,
            "max_similarity": max_similarity,
            "avg_similarity": avg_similarity
        }
        
        # Determine redundancy
        is_redundant = max_similarity >= SIMILARITY_THRESHOLD
        
        # Get top similar papers
        top_similar_papers = similar_papers[:5]
        
        return {
            "is_redundant": is_redundant,
            "reason": f"Max similarity: {max_similarity:.3f} (threshold: {SIMILARITY_THRESHOLD})",
            "max_similarity": max_similarity,
            "similarity_distribution": similarity_distribution,
            "top_similar_papers": top_similar_papers,
            "most_similar_paper": top_similar_papers[0].get('title', 'Unknown') if top_similar_papers else 'None'
        }
        
    except Exception as e:
        logger.error(f"Error in paper similarity analysis: {e}")
        return {
            "is_redundant": False,
            "reason": f"Error during similarity analysis: {str(e)}",
            "max_similarity": 0.0,
            "similarity_distribution": {},
            "top_similar_papers": []
        }

# Legacy exact matching (kept as fallback)
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

# Lambda Handler
def lambda_handler(event, context):
    ensure_index()
    
    # Add embeddings to existing papers on first run
    add_embeddings_to_existing_papers()
    
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
            
            # First check for exact duplicates (legacy)
            if is_duplicate(title_norm, sha_abs):
                logger.info(f"Exact duplicate skipped | {title}")
                continue

            # RAG-based redundancy check
            logger.info(f"Performing RAG-based redundancy check for: {title}")
            redundancy_check = is_paper_redundant_rag(title, abstract)
            
            if redundancy_check["is_redundant"]:
                logger.info(f"Paper rejected as redundant | {title} | {redundancy_check['reason']}")
                logger.info(f"Most similar paper: {redundancy_check.get('most_similar_paper', 'Unknown')}")
                continue
            else:
                logger.info(f"Paper passed RAG redundancy check | {title} | Max similarity: {redundancy_check['max_similarity']:.3f}")

            # Evaluate with Claude (Bedrock)
            # evaluation = evaluate_paper_with_claude(title, abstract)
            # relevance = evaluation.get("relevance", "unknown").lower()
            # novelty = evaluation.get("novelty", "unknown").lower()
            # reason = evaluation.get("reason", "No reason provided.")
            relevance = 'yes'
            novelty = 'yes'

            # Only store relevant + novel papers
            if relevance == "yes" and novelty == "yes":
                # Generate embedding for the paper
                embedding = generate_embedding(abstract)
                
                doc = {
                    "title": title,
                    "title_normalized": title_norm,
                    "authors": authors,
                    "abstract": abstract,
                    "abstract_embedding": embedding,  # Add vector embedding
                    "date": date.replace("/", "-") if date else None,
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "sha_abstract": sha_abs,
                    "decision": "accept",
                    "reason": "RAG-based evaluation passed",
                    "relevance": relevance,
                    "novelty": novelty,
                    "similarity_score": redundancy_check["max_similarity"],
                    "most_similar_paper": redundancy_check.get("most_similar_paper", ""),
                    "ingested_at": int(time.time() * 1000)
                }
                os_client.index(index=OPENSEARCH_INDEX, body=doc, refresh=True)
                logger.info(f"Paper accepted and indexed with RAG | {title} | Similarity: {redundancy_check['max_similarity']:.3f}")
            else:
                logger.info(f"Skipped (irrelevant or not novel) | {title}")

        except Exception as e:
            logger.exception(f"Failed to process message {msg_id}: {e}")
            failures.append({"itemIdentifier": msg_id})

    logger.info(f"Lambda complete. Failures: {len(failures)}")
    return {"batchItemFailures": failures}
