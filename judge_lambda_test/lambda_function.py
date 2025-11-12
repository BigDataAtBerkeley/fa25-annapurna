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
                    # Exponential backoff: 2^attempt + random jitter (0-1 seconds)
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
                    "k": size + 1 if exclude_id else size  # Request extra if filtering
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
            
            # Skip the document we're testing (self-comparison guard)
            if exclude_id and doc_id == exclude_id:
                logger.info(f"Skipping self-match: {doc_id}")
                continue
            
            src = hit.get('_source', {})
            src['_id'] = doc_id
            src['similarity_score'] = hit.get('_score')
            out.append(src)
            
            # Stop once we have enough results
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
        return {"is_redundant": False, "reason": "No similar papers found", "max_similarity": 0.0}
    
    most_similar = similar[0]
    max_sim = most_similar.get('similarity_score', 0.0) or 0.0
    logger.info(f"Most similar paper has similarity {max_sim:.3f} (threshold={SIMILARITY_THRESHOLD})")
    
    rag_context_window = ""
    
    for i, sim in enumerate(similar, start=1):
        if sim.get('similarity_score', 0.0) < SIMILARITY_THRESHOLD:
            break
        rag_context_window += f"\n[{i}] Title: {sim.get('title', 'Unknown')}\nAbstract: {sim.get('abstract', 'No abstract')}"

    return rag_context_window
    """
    return {
        "is_redundant": max_sim >= SIMILARITY_THRESHOLD,
        "reason": f"Most similar has {max_sim:.3f} similarity (threshold={SIMILARITY_THRESHOLD})",
        "max_similarity": max_sim,
        "most_similar_paper": most_similar.get('title', 'Unknown'),
        "most_similar_id": most_similar.get('_id', 'Unknown')
    }
    """

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
        
        # Add exclusion filter if provided
        if exclude_id:
            exact_query["query"]["bool"]["must_not"] = [
                {"term": {"_id": exclude_id}}
            ]

        res = os_client.search(index=OPENSEARCH_INDEX, body=exact_query)
        return res["hits"]["total"]["value"] > 0
    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")
        return False

def evaluate_paper_with_claude(title: str, abstract: str, rag_context_window: str, max_retries: int = 6) -> Dict[str, str]:
    rag_prompt_section = ""
    if rag_context_window:
        rag_prompt_section = f"""
    SIMILARITY — evaluate how closely this paper replicates or overlaps with previously indexed work.
    Use the following retrieved papers as reference context:

    {rag_context_window}

    Say "yes" to SIMILARITY if the current paper is **near-identical** in method, architecture, or result claims to one or more of the retrieved papers,
    without offering a substantial new improvement, extension, or validation.
    Say "no" if the paper is about a distinct topic, or makes a clear, material improvement (e.g., better results, new component, novel analysis, or broader applicability).
    Say "unclear" only if the overlap cannot be confidently determined from the abstract.
    """

    prompt = f"""
    You are an expert ML research analyst embedded in an AWS AI automation pipeline.

    Decide whether this paper is:
    1) RELEVANT to current LLM/AI/ML work,
    2) NOVEL beyond the state of the art,
    3) COMPATIBLE with AWS Trainium for practical implementation,
    {('4) DISTINCT enough from previously indexed work (SIMILARITY check).' if rag_context_window else '').strip()}
    
    Skew your judgements towards being conservative in acceptance, prioritizing papers that clearly push the field forward
    with implementable techniques that would provide valuable insights by testing performance on Trainium hardware.

    Use this rubric:

    RELEVANCE — say "yes" only if the paper is clearly about one or more of:
    - model architectures (LLMs, Transformers, CNNs, GNNs, diffusion models, MoE, hybrid or modular systems, etc.),
    - training algorithms or optimization strategies (optimizers, regularization, loss design, curriculum learning, meta-learning, etc.),
    - efficiency improvements (training or inference speedups, quantization, sparsity, parallelism, mixed precision, distributed training, etc.),
    - alignment, fine-tuning, or data-centric techniques (RLHF, DPO/ORPO, synthetic data, augmentation, retrieval, multimodal alignment, etc.)
    The paper must implement or propose new, directly implementable ML techniques, not just analyze or survey existing ones.
    Say "no" if the paper is clearly about one or more of:
    - survey/position/ethics/policy piece,
    - a benchmark proposal or method of evaluation,
    - an application of an existing model to a domain without new ML techniques,
    - non-neural stats unrelated to MLs or evaluation/safety methods
    - purely theoretical work without clear implementation.

    NOVELTY — judge relative to widely known 2024–2025 techniques (e.g., Llama-3/Mistral/Gemma/Claude-class practices).
    - novel = "yes" if it proposes a new algorithm/architecture or materially better training/inference method with credible evidence
    (theory or strong experiments), not just a dataset swap or minor tuning.
    - novel = "no" if it is mostly packaging/ablations/parameter tweaks/benchmarking of known tricks.

    TRAINIUM COMPATIBILITY — respond "yes" only if the method can realistically run on Trainium:
    - Expressible in PyTorch/XLA; uses FP16/BF16; relies on transformer-compatible ops or ops with XLA kernels (e.g., Flash-Attention-style
    kernels that have XLA paths); no proprietary hardware requirements (TPU-only) or CUDA-only custom kernels without XLA equivalents.
    - If the abstract lacks enough detail to judge, respond "unclear". If it depends on unavailable kernels or CUDA-only custom ops, respond "no".

    {rag_prompt_section}

    Edge rules:
    - Surveys/position/ethics/benchmark-only ⇒ relevance="no", novelty="no".
    - If the abstract is too vague to tell, set novelty="no" and trainium_compatibility="unclear".
    - Keep justification short and cite the decisive claim(s) from the abstract.

    Paper:
    Title: {title}
    Abstract: {abstract}

    Respond with STRICT JSON only, exactly:
    {{
    "relevance": "yes" | "no",
    "novelty": "yes" | "no",
    "trainium_compatibility": "yes" | "no" | "unclear",
    "similarity": {"\"yes\" | \"no\" | \"unclear\"" if rag_context_window else "\"no\""},
    "reason": "<≤3 short sentences citing decisive factors from the abstract and RAG context if applicable>"
    }}
    """.strip()
    
    logger.info(f"Similar papers prompt: {rag_prompt_section}")
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 200,
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
                    # Exponential backoff: 2^attempt + random jitter (0-1 seconds)
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Claude throttled on '{title[:50]}...', retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Claude throttled after {max_retries} attempts for '{title[:50]}...'")
                    return {
                        "relevance": "unknown", 
                        "novelty": "unknown", 
                        "similarity": "unknown",
                        "trainium_compatibility": "unknown",
                        "reason": "Bedrock throttling - max retries exceeded"
                    }
            else:
                logger.warning(f"Claude via Bedrock failed: {e}")
                return {
                    "relevance": "unknown", 
                    "novelty": "unknown", 
                    "similarity": "unknown",
                    "trainium_compatibility": "unknown",
                    "reason": f"Claude evaluation failed: {str(e)}"
                }
    
    return {"relevance": "unknown", "novelty": "unknown", "similarity": "unknown", "trainium_compatibility": "unknown", "reason": "Unexpected error in retry loop"}

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
    paper_id = body.get("paper_id")  # For self-comparison guard
    
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
    
    # Exact duplicate pre-check (excluding self)
    if is_duplicate(title_norm, sha_abs, exclude_id=paper_id):
        reason = "Exact duplicate by title_normalized or sha_abstract"
        result.update({
            "decision": "reject",
            "rejected_by": "exact_duplicate",
            "reason": reason,
            "relevance": "unknown",
            "novelty": "unknown"
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
            "novelty": "unknown"
        })
        return result
    
    # RAG redundancy pre-check - use the pre-generated embedding
    logger.info(f"Retrieving RAG context window redundancy for: {title[:50]}...")
    rag_context_window = retrieve_rag_context_window(title, abstract, paper_embedding, exclude_id=paper_id)
    
    """
    if redundancy.get("is_redundant"):
        reason = redundancy.get("reason", "RAG redundancy")
        result.update({
            "decision": "reject",
            "rejected_by": "rag",
            "reason": reason,
            "max_similarity": redundancy.get("max_similarity"),
            "most_similar_paper": redundancy.get("most_similar_paper"),
            "most_similar_id": redundancy.get("most_similar_id"),
            "relevance": "unknown",
            "novelty": "unknown"
        })
        logger.info(f"[SIMULATION] Rejected by RAG | {title} | {reason}")
        return result
    """    
    
    # Evaluate with Claude (Bedrock)
    evaluation = evaluate_paper_with_claude(title, abstract, rag_context_window)
    relevance = evaluation.get("relevance", "unknown").lower()
    similarity = evaluation.get("similarity", "unknown").lower()
    trainium_compatibility = evaluation.get("trainium_compatibility", "unknown").lower()
    novelty = evaluation.get("novelty", "unknown").lower()
    reason = evaluation.get("reason", "No reason provided.")
    
    result.update({
        "relevance": relevance,
        "novelty": novelty,
        "similarity": similarity,
        "trainium_compatibility": trainium_compatibility,
        "claude_reason": reason
    })
    
    # Only accept relevant + novel papers
    if relevance == "yes" and novelty == "yes" and similarity != "yes" and trainium_compatibility != "no":
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