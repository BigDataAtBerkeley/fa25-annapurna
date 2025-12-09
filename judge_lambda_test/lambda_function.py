# This is a simulation version of the judge lambda that does not modify any data in OpenSearch.
# Requests are sent using setup_test.py and require existing documents in OpenSearch.

import boto3
import json
import os
import time
import hashlib
import logging
import certifi
import ssl
import random
from typing import List, Dict
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['AWS_CA_BUNDLE'] = certifi.where()

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info("Judge Lambda Simulation started")

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers")
RESULTS_QUEUE_URL = os.getenv("RESULTS_QUEUE_URL")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

# AWS Clients
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
sqs_client = boto3.client("sqs", region_name=AWS_REGION)

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

def generate_embedding(text: str, max_retries: int = 6) -> List[float]:
    """Generate embedding with exponential backoff retry logic for throttling."""
    body = json.dumps({"inputText": text})
    
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=body,
                contentType="application/json"
            )
            result = json.loads(response["body"].read())
            return result.get("embedding", [])
            
        except Exception as e:
            error_str = str(e)
            if "ThrottlingException" in error_str or "Too many requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Titan embedding throttled, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Titan embedding throttled after {max_retries} attempts")
                    return []
            else:
                logger.warning(f"Embedding generation failed: {e}")
                return []
    
    return []

def search_similar_papers_with_embedding(
    embedding: List[float], 
    exclude_id: str = None, 
    size: int = 5
) -> List[Dict]:
    """
    Perform KNN search using a pre-generated embedding.
    Avoids redundant embedding generation.
    """
    try:
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
                    "k": size + 1 if exclude_id else size
                }
            }
        }
        
        try:
            res = os_client.search(
                index=OPENSEARCH_INDEX, 
                body={"query": query_primary, "size": size + 1 if exclude_id else size}
            )
        except Exception as e1:
            logger.warning(f"Primary kNN query failed, retrying with field/query_vector: {e1}")
            query_fallback = {
                "knn": {
                    "field": "abstract_embedding",
                    "query_vector": embedding,
                    "k": size + 1 if exclude_id else size
                }
            }
            res = os_client.search(
                index=OPENSEARCH_INDEX, 
                body={"query": query_fallback, "size": size + 1 if exclude_id else size}
            )
        
        out = []
        for hit in res.get('hits', {}).get('hits', []):
            doc_id = hit.get('_id')
            
            if exclude_id and doc_id == exclude_id:
                logger.info(f"Skipping self-match: {doc_id}")
                continue
            
            src = hit.get('_source', {})
            src['_id'] = doc_id
            src['similarity_score'] = hit.get('_score')
            out.append(src)
            
            if len(out) >= size:
                break
                
        return out
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        return []

def retrieve_rag_context_window(
    title: str, 
    abstract: str, 
    embedding: List[float],
    exclude_id: str = None
) -> str:
    
    similar = search_similar_papers_with_embedding(embedding, exclude_id=exclude_id, size=5)
    logger.info(f"Found {len(similar)} similar papers for RAG redundancy check.")
    
    if not similar:
        logger.info(f"[CRITICAL FLAG] No similar papers for RAG redundancy check.")
        return ""
    
    most_similar = similar[0]
    max_sim = most_similar.get('similarity_score', 0.0) or 0.0
    logger.info(f"Most similar paper has similarity {max_sim:.3f} (threshold={SIMILARITY_THRESHOLD})")
    
    rag_context_window = ""
    
    for i, sim in enumerate(similar, start=1):
        if sim.get('similarity_score', 0.0) < SIMILARITY_THRESHOLD:
            break
        rag_context_window += f"\n[{i}] Title: {sim.get('title', 'Unknown')}\nAbstract: {sim.get('abstract', 'No abstract')}"

    return rag_context_window

def is_duplicate(title_norm: str, sha_abs: str, exclude_id: str = None) -> bool:
    """
    Check for exact duplicates by title or abstract hash.
    Excludes the paper itself from comparison in simulation mode.
    """
    try:
        query_filters = [
            {"term": {"title_normalized": title_norm}},
            {"term": {"sha_abstract": sha_abs}}
        ]
        
        exact_query = {
            "query": {
                "bool": {
                    "should": query_filters,
                    "minimum_should_match": 1
                }
            }
        }
        
        if exclude_id:
            exact_query["query"]["bool"]["must_not"] = [
                {"term": {"_id": exclude_id}}
            ]

        res = os_client.search(index=OPENSEARCH_INDEX, body=exact_query)
        return res["hits"]["total"]["value"] > 0
    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")
        return False

def evaluate_initial_gates(title: str, abstract: str, max_retries: int = 6) -> Dict[str, str]:
    prompt = f"""
You are an expert ML research analyst embedded in an AWS AI automation pipeline.
Be *conservative* in all judgments - If anything is vague, ambiguous, or not clearly implementable, choose "no" or "unclear".
Only chose "yes" if the paper would be valuable to implement and train on AWS Trainium in 2025.
Evaluate this paper across two dimensions: RELEVANCE and TRAINIUM COMPATIBILITY.

--------------------------------------------------------------------------------
RELEVANCE (strict)
--------------------------------------------------------------------------------
Say “yes” ONLY if the abstract does ALL of the following:
- directly targets *training* or *inference* for language or vision models (Transformers, Diffusion models, CNNs,etc.), AND
- proposes a concrete, implementable architectural or algorithmic innovation, AND
- the innovation materially affects model training, scaling laws, optimization strategies, fine-tuning, or inference efficiency.

Examples of relevant contributions:
- New model architectures (e.g., novel Transformer variants, diffusion architectures, new MoE routing designs, hybrid or modular systems)
- New training algorithms or optimization techniques (e.g., new optimizers, regularization strategies, loss functions, curriculum learning, meta-learning, scalable pretraining procedures)
- Efficiency or scalability improvements (e.g., faster attention mechanisms, quantization schemes, sparsity methods, parallelism or distributed training strategies, mixed-precision or memory-optimized kernels)

Say if "no" if any of the following are true:
- the abstract is vague or uses high-level non-technical language,
- missing specific details on the model, architecture, or algorithm,
- the paper focuses on methods of improvement outside the scope of training or inference, such as dataset curation, synthetic data generation, deployment of agents, or specialized prompting etc.
- the paper is purely empirical: benchmarks, datasets, metrics, evaluations, surveys, ablations, or analysis of existing techniques,
- the paper is an application of existing models to a domain (e.g., medicine, law, robotics, finance) without core training/architecture innovation,
- the contributions are incremental or trivial (dataset swaps, hyperparameter tweaks, small add-on modules, minor architecture variations),
- theoretical work without a clear implementation path.

Examples of irrelevant contributions:
- A paper that proposes a new visual-audio dataset that improves the quality of image-to-audio generation.
- A paper that focuses on analyzing internal embedding representations of hallucinations in language models.
- A paper that researhes the affects of deplyoing multiple agents within a single enviorment.
- A paper that creates a novel method of chain-of-thought reasoning.
- A paper that proposes a new evaluation process for different metrics of reasoning abilities.

If uncertain, set relevance= "no".

--------------------------------------------------------------------------------
TRAINIUM COMPATIBILITY (not as strict as the initial evaluation)
--------------------------------------------------------------------------------
Say “yes” ONLY if the paper’s method is clearly implementable using PyTorch/XLA with
available kernels or transformer-compatible operations.

Say “no” if:
- the work relies on custom CUDA kernels, Triton kernels, fused GPU-only ops,
- the method likely requires specialized GPU implementations with no XLA path,
- the architecture includes exotic attention mechanisms or convolutions lacking XLA ops.

If the abstract does not provide enough implementation detail, set "unclear".

--------------------------------------------------------------------------------
Paper:
Title: {title}
Abstract: {abstract}

Respond with STRICT JSON only, exactly:
{{
"relevance": "yes" | "no",
"relevance_reason": "<≤2 short sentences>",
"trainium_compatibility": "yes" | "no" | "unclear",
"trainium_reason": "<≤2 short sentences>"
}}
""".strip()
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 300,
        "temperature": 0.3
    })
    
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=body
            )
            text = json.loads(response["body"].read())["content"][0]["text"].strip()
            return json.loads(text)
            
        except Exception as e:
            error_str = str(e)
            if "ThrottlingException" in error_str or "Too many requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Claude throttled on initial eval '{title[:50]}...', retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Claude initial eval throttled after {max_retries} attempts")
                    return {
                        "relevance": "unknown",
                        "relevance_reason": "Bedrock throttling - max retries exceeded",
                        "trainium_compatibility": "unknown",
                        "trainium_reason": "Bedrock throttling - max retries exceeded"
                    }
            else:
                logger.warning(f"Claude initial eval failed: {e}")
                return {
                    "relevance": "unknown",
                    "relevance_reason": f"Evaluation failed: {str(e)}",
                    "trainium_compatibility": "unknown",
                    "trainium_reason": f"Evaluation failed: {str(e)}"
                }
    
    return {
        "relevance": "unknown",
        "relevance_reason": "Unexpected error in retry loop",
        "trainium_compatibility": "unknown",
        "trainium_reason": "Unexpected error in retry loop"
    }


def evaluate_similarity_with_claude(title: str, abstract: str, rag_context_window: str, max_retries: int = 6) -> Dict[str, str]:
    """
    Evaluate similarity to existing papers using RAG context.
    """
    prompt = f"""
You are an expert ML research analyst embedded in an AWS AI automation pipeline.

Evaluate how closely this paper replicates or overlaps with previously indexed work.

Use the following retrieved papers as reference context:

{rag_context_window}

SIMILARITY — evaluate how closely this paper replicates or overlaps with the retrieved papers.
Say "yes" to SIMILARITY if the current paper is similar in method, architecture, or result claims to one or more of the retrieved papers,
without offering a substantial new improvement, extension, or validation. 
Say "no" if the paper is about a distinct topic, or makes a clear, material improvement (e.g., better results, new component, novel analysis, or broader applicability).
Say "unclear" only if the overlap cannot be confidently determined from the abstract.

Paper to evaluate:
Title: {title}
Abstract: {abstract}

Respond with STRICT JSON only, exactly:
{{
"similarity": "yes" | "no" | "unclear",
"reason": "<≤2 short sentences citing decisive factors from the abstract and RAG context. If the similarity was yes, please include the title of the most similar paper.>"
}}
""".strip()
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 150,
        "temperature": 0.3
    })
    
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=body
            )
            text = json.loads(response["body"].read())["content"][0]["text"].strip()
            return json.loads(text)
            
        except Exception as e:
            error_str = str(e)
            if "ThrottlingException" in error_str or "Too many requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Claude throttled on similarity eval '{title[:50]}...', retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Claude similarity eval throttled after {max_retries} attempts")
                    return {"similarity": "unknown", "reason": "Bedrock throttling - max retries exceeded"}
            else:
                logger.warning(f"Claude similarity eval failed: {e}")
                return {"similarity": "unknown", "reason": f"Evaluation failed: {str(e)}"}
    
    return {"similarity": "unknown", "reason": "Unexpected error in retry loop"}


def evaluate_paper_with_claude(title: str, abstract: str, rag_context_window: str, max_retries: int = 6) -> Dict[str, str]:
    """
    Orchestrator function with early termination logic.
    Now uses combined initial evaluation (relevance, trainium) in one call,
    then separate similarity check if needed.
    """
    reasons = []
    
    # Step 1: Combined evaluation of relevance and Trainium compatibility
    logger.info(f"Evaluating relevance and Trainium compatibility for: {title[:50]}...")
    initial_eval = evaluate_initial_gates(title, abstract, max_retries)
    
    relevance = initial_eval.get("relevance", "unknown").lower()
    trainium_compatibility = initial_eval.get("trainium_compatibility", "unknown").lower()
    
    reasons.append(f"Relevance: {initial_eval.get('relevance_reason', 'N/A')}")
    reasons.append(f"Trainium: {initial_eval.get('trainium_reason', 'N/A')}")
    
    # Early termination checks
    if relevance != "yes":
        logger.info(f"Early termination: relevance={relevance}")
        return {
            "relevance": relevance,
            "trainium_compatibility": trainium_compatibility,
            "similarity": "not_evaluated",
            "reason": " | ".join(reasons),
            "early_termination": "relevance"
        }
    
    if trainium_compatibility == "no":
        logger.info(f"Early termination: trainium_compatibility={trainium_compatibility}")
        return {
            "relevance": relevance,
            "trainium_compatibility": trainium_compatibility,
            "similarity": "not_evaluated",
            "reason": " | ".join(reasons),
            "early_termination": "trainium_compatibility"
        }
    
    # Step 2: Check similarity only if RAG context exists and all gates passed
    similarity = "no"
    if rag_context_window:
        logger.info(f"Evaluating similarity for: {title[:50]}...")
        similarity_eval = evaluate_similarity_with_claude(title, abstract, rag_context_window, max_retries)
        similarity = similarity_eval.get("similarity", "unknown").lower()
        reasons.append(f"Similarity: {similarity_eval.get('reason', 'N/A')}")
    else:
        reasons.append("Similarity: No RAG context available")
    
    # All gates passed
    return {
        "relevance": relevance,
        "trainium_compatibility": trainium_compatibility,
        "similarity": similarity,
        "reason": " | ".join(reasons),
        "early_termination": None
    }


def send_result_to_queue(result: Dict) -> bool:
    """Send simulation result to SQS results queue."""
    if not RESULTS_QUEUE_URL:
        logger.warning("RESULTS_QUEUE_URL not set, skipping queue send")
        return False
    
    try:
        sqs_client.send_message(
            QueueUrl=RESULTS_QUEUE_URL,
            MessageBody=json.dumps(result)
        )
        logger.info(f"Sent result to queue: {result.get('title', 'Unknown')}")
        return True
    except Exception as e:
        logger.error(f"Failed to send result to queue: {e}")
        return False

def simulate_paper_evaluation(body: Dict, msg_id: str) -> Dict:
    """
    Simulate the entire judge lambda evaluation pipeline without modifying data.
    Returns a result dictionary with the decision and reasoning.
    """
    title = body["title"]
    abstract = body.get("abstract", "")
    authors = body.get("authors", [])
    s3_bucket = body.get("s3_bucket", "")
    s3_key = body.get("s3_key", "")
    date = body.get("date")
    paper_id = body.get("paper_id")
    
    title_norm = normalize_title(title)
    sha_abs = sha256_text(abstract)
    
    result = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "date": date,
        "message_id": msg_id,
        "evaluated_at": datetime.utcnow().isoformat()
    }
    
    # Exact duplicate pre-check
    if is_duplicate(title_norm, sha_abs, exclude_id=paper_id):
        reason = "Exact duplicate by title_normalized or sha_abstract"
        result.update({
            "decision": "reject",
            "rejected_by": "exact_duplicate",
            "reason": reason,
            "relevance": "unknown",
        })
        logger.info(f"[SIMULATION] Skipped (exact duplicate) | {title} | {reason}")
        return result
    
    logger.info(f"Generating embedding for: {title[:50]}...")
    paper_embedding = generate_embedding(abstract)
    
    if not paper_embedding:
        logger.error(f"Failed to generate embedding for: {title}")
        result.update({
            "decision": "error",
            "rejected_by": "embedding_failure",
            "reason": "Could not generate embedding",
            "relevance": "unknown",
        })
        return result
    
    # RAG context retrieval
    logger.info(f"Retrieving RAG context window for: {title[:50]}...")
    rag_context_window = retrieve_rag_context_window(title, abstract, paper_embedding, exclude_id=paper_id)
    
    # Evaluate with Claude (now condensed to 2 calls instead of 4)
    evaluation = evaluate_paper_with_claude(title, abstract, rag_context_window)
    relevance = evaluation.get("relevance", "unknown").lower()
    similarity = evaluation.get("similarity", "unknown").lower()
    trainium_compatibility = evaluation.get("trainium_compatibility", "unknown").lower()
    reason = evaluation.get("reason", "No reason provided.")
    
    result.update({
        "relevance": relevance,
        "similarity": similarity,
        "trainium_compatibility": trainium_compatibility,
        "claude_reason": reason
    })
    
    # Only accept relevant + novel papers
    if relevance == "yes" and similarity == "no" and trainium_compatibility != "no":
        result.update({
            "decision": "accept",
            "rejected_by": None,
            "reason": reason
        })
        logger.info(f"[SIMULATION] Would accept | {title}")
    else:
        result.update({
            "decision": "reject",
            "rejected_by": "claude",
            "reason": reason
        })
        logger.info(f"[SIMULATION] Rejected by Claude | {title} | {reason}")
    
    return result

# Lambda Handler
def lambda_handler(event, context):
    logger.info("Starting simulation lambda handler")
    failures = []
    results = []

    for record in event.get("Records", []):
        msg_id = record["messageId"]

        try:
            body = json.loads(record["body"])
            
            # Simulate the evaluation
            result = simulate_paper_evaluation(body, msg_id)
            results.append(result)
            
            # Send result to queue
            send_result_to_queue(result)
            
            # Gentle rate limiting between papers
            time.sleep(1)
            
        except Exception as e:
            logger.exception(f"Failed to process message {msg_id}: {e}")
            failures.append({"itemIdentifier": msg_id})

    logger.info(f"Simulation complete. Processed: {len(results)}, Failures: {len(failures)}")
    return {
        "batchItemFailures": failures,
        "simulation_results": results
    }