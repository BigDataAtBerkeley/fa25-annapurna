import boto3
import json
import os
import time
import hashlib
import logging
import random
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

def generate_embedding(text: str, max_retries: int = 6) -> List[float]:
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

def evaluate_initial_gates(title: str, abstract: str, max_retries: int = 6) -> Dict[str, str]:
    """
    Combined evaluation of relevance, novelty, and Trainium compatibility in a single LLM call.
    """
    prompt = f"""
You are an expert ML research analyst embedded in an AWS AI automation pipeline. 
Be conservative in your judgements - if the abstract is vague, lean towards "no" or "unclear". 
Only chose "yes" if the paper would be valuable to implement and train on AWS Trainium in 2025.

Evaluate this paper across three dimensions: RELEVANCE, NOVELTY, and TRAINIUM COMPATIBILITY.

RELEVANCE — say "yes" only if the paper:
- proposes an innovative, new model architecture (LLMs, Transformers, CNNs, GNNs, diffusion models, MoE, hybrid or modular systems, etc.),
- training algorithms or optimization strategies (optimizers, regularization, loss design, curriculum learning, meta-learning, etc.),
- efficiency improvements (training or inference speedups, quantization, sparsity, parallelism, mixed precision, distributed training, etc.),
- alignment, fine-tuning, or data-centric techniques (RLHF, DPO/ORPO, synthetic data, augmentation, retrieval, multimodal alignment, etc.)

The paper must implement or propose new, directly implementable ML techniques, not just analyze or survey existing ones.

Say "no" if the paper:
- is a survey/position/ethics/policy piece or new analysis of existing methods,
- proposes only benchmarks or methods of evaluation,
- an application of an existing model or LLMs to a specific domain (e.g., healthcare, finance) without revisions to the training or architecture,
- makes trivial modifications to existing models, such as hyperparameter tuning, dataset swaps, or adding singular layers to known architectures,
- non-neural stats unrelated to ML or evaluation/safety methods
- purely theoretical work without clear implementation.

NOVELTY — judge relative to widely known 2024–2025 techniques (e.g., Llama-3/Mistral/Gemma/Claude-class practices).
- novel = "yes" if it proposes a new algorithm/architecture or materially better training/inference method with credible evidence
(theory or strong experiments), not just a dataset swap or minor tuning.
- novel = "no" if it is mostly packaging/ablations/parameter tweaks/benchmarking of known tricks. Remember, a new benchmark is NOT considered novel. If the paper is merely about a new benchmark, it is not novel.

If the abstract is too vague to tell, set novelty="no".

TRAINIUM COMPATIBILITY — respond "yes" only if the method can realistically run on AWS Trainium:
- Expressible in PyTorch/XLA; uses FP16/BF16; relies on transformer-compatible ops or ops with XLA kernels (e.g., Flash-Attention-style
kernels that have XLA paths); no proprietary hardware requirements (TPU-only) or CUDA-only custom kernels without XLA equivalents.
- If the abstract lacks enough detail to judge, respond "unclear". If it depends on unavailable kernels or CUDA-only custom ops, respond "no".

Paper:
Title: {title}
Abstract: {abstract}

Respond with STRICT JSON only, exactly:
{{
"relevance": "yes" | "no",
"relevance_reason": "<≤2 short sentences>",
"novelty": "yes" | "no",
"novelty_reason": "<≤2 short sentences>",
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
                        "novelty": "unknown",
                        "novelty_reason": "Bedrock throttling - max retries exceeded",
                        "trainium_compatibility": "unknown",
                        "trainium_reason": "Bedrock throttling - max retries exceeded"
                    }
            else:
                logger.warning(f"Claude initial eval failed: {e}")
                return {
                    "relevance": "unknown",
                    "relevance_reason": f"Evaluation failed: {str(e)}",
                    "novelty": "unknown",
                    "novelty_reason": f"Evaluation failed: {str(e)}",
                    "trainium_compatibility": "unknown",
                    "trainium_reason": f"Evaluation failed: {str(e)}"
                }
    
    return {
        "relevance": "unknown",
        "relevance_reason": "Unexpected error in retry loop",
        "novelty": "unknown",
        "novelty_reason": "Unexpected error in retry loop",
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


# Claude via Bedrock evaluation
def evaluate_paper_with_claude(title: str, abstract: str, rag_context_window: str, max_retries: int = 6) -> Dict[str, str]:
    """
    Orchestrator function with early termination logic.
    Now uses combined initial evaluation (relevance, novelty, trainium) in one call,
    then separate similarity check if needed.
    """
    reasons = []
    
    # Step 1: Combined evaluation of relevance, novelty, and Trainium compatibility
    logger.info(f"Evaluating relevance, novelty, and Trainium compatibility for: {title[:50]}...")
    initial_eval = evaluate_initial_gates(title, abstract, max_retries)
    
    relevance = initial_eval.get("relevance", "unknown").lower()
    novelty = initial_eval.get("novelty", "unknown").lower()
    trainium_compatibility = initial_eval.get("trainium_compatibility", "unknown").lower()
    
    reasons.append(f"Relevance: {initial_eval.get('relevance_reason', 'N/A')}")
    reasons.append(f"Novelty: {initial_eval.get('novelty_reason', 'N/A')}")
    reasons.append(f"Trainium: {initial_eval.get('trainium_reason', 'N/A')}")
    
    # Early termination checks
    if relevance != "yes":
        logger.info(f"Early termination: relevance={relevance}")
        return {
            "relevance": relevance,
            "novelty": novelty,
            "trainium_compatibility": trainium_compatibility,
            "similarity": "not_evaluated",
            "reason": " | ".join(reasons),
            "early_termination": "relevance"
        }
    
    if novelty != "yes":
        logger.info(f"Early termination: novelty={novelty}")
        return {
            "relevance": relevance,
            "novelty": novelty,
            "trainium_compatibility": trainium_compatibility,
            "similarity": "not_evaluated",
            "reason": " | ".join(reasons),
            "early_termination": "novelty"
        }
    
    if trainium_compatibility == "no":
        logger.info(f"Early termination: trainium_compatibility={trainium_compatibility}")
        return {
            "relevance": relevance,
            "novelty": novelty,
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
        "novelty": novelty,
        "trainium_compatibility": trainium_compatibility,
        "similarity": similarity,
        "reason": " | ".join(reasons),
        "early_termination": None
    }

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
            paper_id = body.get("paper_id")
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
            
            message_attributes = record.get('attributes', {})
            receive_count = message_attributes.get('ApproximateReceiveCount', '1')
            logger.info(f"Processing {title} - Receive count: {receive_count}")
            
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
                #index_paper_document(reject_doc)
                logger.info(f"Skipped (exact duplicate) | {title} | {reason}")
                continue
            
            rag_context_window = retrieve_rag_context_window(title, abstract, precomputed_embedding, paper_id)

            # Evaluate with Claude (Bedrock)
            evaluation = evaluate_paper_with_claude(title, abstract, rag_context_window)
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
                logger.info(f"Skipped (irrelevant or not novel) | {title} | {reason}")

        except Exception as e:
            logger.exception(f"Failed to process message {msg_id}: {e}")
            failures.append({"itemIdentifier": msg_id})

    logger.info(f"Lambda complete. Failures: {len(failures)}")
    return {"batchItemFailures": failures}
