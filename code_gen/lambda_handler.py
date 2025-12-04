"""
Lambda handler for Pipeline Code Generation.

This Lambda function:
1. Generates code using ChunkedPyTorchGenerator (PDF chunking with smart relevance filtering)
2. Sends initial generated code to the async Flask app's /execute endpoint for Trainium execution
3. Shuts down – all execution, code review, and retries happen in the async Flask app

This is the entry point for the pipeline Lambda. The Flask app handles the rest of the pipeline.
"""

import os
import json
import logging
import time
import requests
import boto3
from datetime import datetime
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError
import sys

# Set LD_LIBRARY_PATH to include system library paths (for libcrypt.so.2 and other shared libs)
# For container-based Lambda, libraries are in standard Linux locations
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
# Add common library paths for container Lambda
library_paths = [
    '/lib64',
    '/usr/lib64', 
    '/var/lang/lib64',
    '/usr/lib',
    '/lib',
    '/var/task',  # In case pymupdf has bundled libs
]
os.environ['LD_LIBRARY_PATH'] = ':'.join(library_paths) + (':' + current_ld_path if current_ld_path else '')

# Lambda layers are automatically added to sys.path, but ensure /opt/python is there
if '/opt/python' not in sys.path:
    sys.path.insert(0, '/opt/python')
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Debug: Check if layer is accessible and test import (before importing modules that need pymupdf)
logger.info("=== Lambda Layer Debug ===")
if os.path.exists('/opt/python'):
    try:
        layer_contents = os.listdir('/opt/python')[:10]
        logger.info(f"✅ /opt/python exists. Contents: {layer_contents}")
        if os.path.exists('/opt/python/fitz'):
            logger.info("✅ /opt/python/fitz directory found")
            # Try to import fitz directly from layer
            try:
                import fitz
                logger.info(f"✅ SUCCESS: fitz imported from layer! Version: {fitz.version if hasattr(fitz, 'version') else 'unknown'}")
            except ImportError as e:
                logger.error(f"❌ FAILED to import fitz from layer: {e}")
            except Exception as e:
                logger.error(f"❌ ERROR importing fitz: {e}")
        else:
            logger.warning("⚠️ /opt/python/fitz directory NOT found")
    except Exception as e:
        logger.warning(f"⚠️ Error checking layer: {e}")
else:
    logger.warning("⚠️ /opt/python does not exist - Lambda layer may not be mounted")
logger.info("=== End Lambda Layer Debug ===")

from chunked_generator import ChunkedPyTorchGenerator  # type: ignore

# Import SlackNotifier for initial paper notifications
try:
    from slack_notifier import SlackNotifier
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logger.warning("slack_notifier module not available - initial paper notifications will be disabled")

# Flask app endpoint for code execution
FLASK_EXECUTE_ENDPOINT = os.getenv('FLASK_EXECUTE_ENDPOINT')  # e.g., "http://1.2.3.4:8000/execute" (deprecated - use queue)
TRAINIUM_EXECUTION_QUEUE_URL = os.getenv('TRAINIUM_EXECUTION_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo')  # SQS queue for execution
USE_QUEUE_FOR_EXECUTION = os.getenv('USE_QUEUE_FOR_EXECUTION', 'true').lower() == 'true'  # Use queue instead of HTTP
ENABLE_EXECUTION_TESTING = os.getenv('ENABLE_EXECUTION_TESTING', 'false').lower() == 'true'

def enqueue_paper_for_execution(paper_id: str, code: str, paper_title: Optional[str] = None, 
                                slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
    """
    Enqueue a paper to the Trainium execution queue.
    
    Args:
        paper_id: Paper ID
        code: Generated code
        paper_title: Paper title (optional)
        slack_thread_ts: Slack thread timestamp (optional)
        
    Returns:
        Result dictionary
    """
    if not TRAINIUM_EXECUTION_QUEUE_URL:
        logger.warning("TRAINIUM_EXECUTION_QUEUE_URL not set - cannot enqueue paper")
        return {
            "success": False,
            "error": "TRAINIUM_EXECUTION_QUEUE_URL not configured"
        }
    
    sqs_client = boto3.client('sqs', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    message_body = {
        "paper_id": paper_id,
        "code": code
    }
    
    if paper_title:
        message_body["paper_title"] = paper_title
    if slack_thread_ts:
        message_body["slack_thread_ts"] = slack_thread_ts
    
    try:
        response = sqs_client.send_message(
            QueueUrl=TRAINIUM_EXECUTION_QUEUE_URL,
            MessageBody=json.dumps(message_body),
            MessageGroupId=paper_id,  # FIFO queue requires MessageGroupId
            MessageDeduplicationId=f"{paper_id}-{int(time.time())}"  # Unique ID for deduplication
        )
        logger.info(f"✅ Enqueued paper {paper_id} to execution queue")
        return {
            "success": True,
            "message_id": response.get('MessageId'),
            "queue_url": TRAINIUM_EXECUTION_QUEUE_URL
        }
    except ClientError as e:
        logger.error(f"Failed to enqueue paper {paper_id}: {e}")
        return {
            "success": False,
            "error": f"Failed to enqueue paper: {str(e)}"
        }


def send_papers_to_trainium_batch(papers: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
    """
    Send multiple papers to Trainium in batches concurrently.
    
    Args:
        papers: List of paper dicts with 'paper_id', 'code', 'paper_title', 'slack_thread_ts'
        batch_size: Maximum papers to send in one batch (default: 10)
        
    Returns:
        Dictionary with batch sending results
    """
    if not papers:
        return {"success": True, "sent": 0}
    
    # Split into batches
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]
    total_sent = 0
    total_failed = 0
    
    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"Sending batch {batch_num}/{len(batches)} to Trainium ({len(batch)} papers)")
        
        # Enqueue all papers in batch
        for paper in batch:
            try:
                if USE_QUEUE_FOR_EXECUTION:
                    # Use queue-based execution
                    result = enqueue_paper_for_execution(
                        paper['paper_id'],
                        paper['code'],
                        paper.get('paper_title'),
                        paper.get('slack_thread_ts')
                    )
                else:
                    # Use HTTP-based execution (legacy)
                    result = send_to_flask_app(
                        paper['paper_id'],
                        paper['code'],
                        paper.get('paper_title'),
                        paper.get('slack_thread_ts')
                    )
                
                if result.get('success'):
                    total_sent += 1
                    logger.info(f"✅ {'Enqueued' if USE_QUEUE_FOR_EXECUTION else 'Sent'} {paper['paper_id']} to Trainium")
                else:
                    total_failed += 1
                    logger.error(f"❌ Failed to {'enqueue' if USE_QUEUE_FOR_EXECUTION else 'send'} {paper['paper_id']}: {result.get('error')}")
            except Exception as e:
                total_failed += 1
                logger.error(f"❌ Error processing {paper['paper_id']}: {e}")
    
    logger.info(f"Batch sending complete: {total_sent} sent, {total_failed} failed")
    return {
        "success": total_failed == 0,
        "sent": total_sent,
        "failed": total_failed,
        "total": len(papers)
    }


def send_to_flask_app(paper_id: str, code: str, paper_title: Optional[str] = None, 
                      slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
    """
    Send reviewed code to Flask app's /execute endpoint for Trainium execution.
    
    Args:
        paper_id: Paper ID
        code: Reviewed code to execute
        paper_title: Paper title (optional)
        slack_thread_ts: Slack thread timestamp (optional)
        
    Returns:
        Response from Flask app
    """
    if not FLASK_EXECUTE_ENDPOINT:
        logger.warning("FLASK_EXECUTE_ENDPOINT not set - cannot send code to Flask app")
        return {
            "success": False,
            "error": "FLASK_EXECUTE_ENDPOINT not configured"
        }
    
    endpoint = FLASK_EXECUTE_ENDPOINT
    if not endpoint.endswith('/execute'):
        endpoint = f"{endpoint.rstrip('/')}/execute"
    
    payload = {
        "paper_id": paper_id,
        "code": code,
        "timeout": int(os.getenv('TRAINIUM_EXECUTION_TIMEOUT', '3600'))  # 60 minutes default
    }
    
    if paper_title:
        payload["paper_title"] = paper_title
    
    if slack_thread_ts:
        payload["slack_thread_ts"] = slack_thread_ts
    
    logger.info(f"Sending reviewed code to Flask app: {endpoint}")
    
    try:
        # Flask app returns 202 Accepted for async execution
        response = requests.post(
            endpoint,
            json=payload,
            timeout=30  # Short timeout - Flask app should accept quickly
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"✅ Code sent to Flask app successfully: {result.get('status', 'accepted')}")
        return {
            "success": True,
            "flask_response": result,
            "status": result.get("status", "accepted"),
            "job_id": result.get("job_id"),
            "status_url": result.get("status_url")
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send code to Flask app: {e}")
        return {
            "success": False,
            "error": f"Failed to send code to Flask app: {str(e)}"
        }


class PipelineHandler:
    """Main handler for the pipeline Lambda (code generation only)."""
    
    def __init__(self):
        """Initialize the pipeline handler."""
        # Initialize code generator (PDF chunking with smart relevance filtering)
        self.generator = ChunkedPyTorchGenerator(
            batch_size=8,  # Group 8 chunk summaries into each batch for hierarchical summarization
            use_smart_pdf_chunking=True,  # Enable smart chunking to prioritize relevant sections (filters appendix)
            max_pdf_chunks=15,  # Maximum number of PDF chunks to process (prioritizes abstract, formulas, diagrams)
            pages_per_pdf_chunk=2  # 2 pages per PDF chunk
        )

        logger.info("Pipeline Handler initialized (code generation only)")
    
    def process_paper(self, paper_id: str, slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single paper: generate code and send it to the async Flask app.
        
        Args:
            paper_id: Paper ID to process
            slack_thread_ts: Optional Slack thread timestamp for notifications
            
        Returns:
            Result dictionary with generation, review, and Flask app response
        """
        start_time = time.time()
        result = {
            "paper_id": paper_id,
            "success": False,
            "pipeline_start": datetime.now().isoformat()
        }
        
        try:
            # Step 0: Send initial paper notification to Slack (creates thread)
            # Use provided slack_thread_ts if available, otherwise create new thread
            if SLACK_AVAILABLE and not slack_thread_ts:
                try:
                    # Get paper from OpenSearch to send initial notification
                    paper = self.generator.opensearch_client.get_paper_by_id(paper_id)
                    if paper:
                        filtered_paper = {k: v for k, v in paper.items() if k != 'embeddings'}
                        filtered_paper['_id'] = paper_id
                        slack_notifier = SlackNotifier()
                        slack_thread_ts = slack_notifier.send_paper_info(filtered_paper)
                        if slack_thread_ts:
                            logger.info(f"✅ Sent initial paper notification to Slack (thread_ts: {slack_thread_ts})")
                            # Store slack_thread_ts in OpenSearch for thread continuity
                            try:
                                self.generator.opensearch_client.client.update(
                                    index=self.generator.opensearch_client.opensearch_index,
                                    id=paper_id,
                                    body={
                                        "doc": {
                                            "slack_thread_ts": slack_thread_ts
                                        }
                                    }
                                )
                                logger.info(f"✅ Stored slack_thread_ts in OpenSearch for {paper_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to store slack_thread_ts in OpenSearch: {e}")
                        else:
                            logger.warning("⚠️ Failed to send initial paper notification to Slack")
                    else:
                        logger.warning(f"Paper {paper_id} not found in OpenSearch - skipping initial Slack notification")
                except Exception as e:
                    logger.warning(f"⚠️ Error sending initial paper notification to Slack: {e}")
            
            # Step 1: Generate code
            # (slack_thread_ts is now set - either from parameter or newly created)
            logger.info(f"Step 1: Generating code for paper {paper_id}...")
            code_gen_result = self.generator.generate_code_for_paper(paper_id)
            
            if not code_gen_result.get("success") or not code_gen_result.get("code"):
                result["error"] = code_gen_result.get("error", "Code generation failed")
                result["code_generation"] = {
                    "success": False,
                    "error": code_gen_result.get("error")
                }
                return result
            
            logger.info(f"✅ Code generated ({len(code_gen_result['code']):,} chars)")
            result["code_generation"] = {
                "success": True,
                "code_length": len(code_gen_result["code"]),
                "model_used": code_gen_result.get("model_used"),
                "recommended_dataset": code_gen_result.get("recommended_dataset")
            }

            # Step 1.5: Send code generation notification to Slack (first follow-up)
            if SLACK_AVAILABLE and slack_thread_ts:
                try:
                    slack_notifier = SlackNotifier()
                    # Get code S3 key if available
                    code_s3_key = None
                    if hasattr(self.generator, 'opensearch_client'):
                        paper = self.generator.opensearch_client.get_paper_by_id(paper_id)
                        if paper:
                            code_s3_key = paper.get('code_s3_key')
                    
                    slack_notifier.send_code_generation_notification(
                        paper_id=paper_id,
                        code_length=len(code_gen_result["code"]),
                        model_used=code_gen_result.get("model_used"),
                        recommended_dataset=code_gen_result.get("recommended_dataset"),
                        code_s3_key=code_s3_key,
                        thread_ts=slack_thread_ts
                    )
                    logger.info(f"✅ Sent code generation notification to Slack")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send code generation notification: {e}")

            # Step 2: Store code for batch sending to Trainium (don't send immediately)
            # Code will be sent in batches after all papers in the batch are processed
            result["code"] = code_gen_result["code"]  # Store code for batch sending
            result["paper_title"] = code_gen_result.get("paper_title")
            result["success"] = True  # Code generation succeeded
            result["pipeline_end"] = datetime.now().isoformat()
            result["total_pipeline_time"] = time.time() - start_time
            
            logger.info(
                f"✅ Code generation complete for paper {paper_id} "
                f"(total time: {result['total_pipeline_time']:.1f}s) - will be sent to Trainium in batch"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            result["error"] = str(e)
            result["pipeline_end"] = datetime.now().isoformat()
            result["total_pipeline_time"] = time.time() - start_time
            return result


def lambda_handler(event, context):
    """
    AWS Lambda handler function for pipeline (code generation only).
    
    This Lambda:
    1. Generates code using ChunkedPyTorchGenerator
    2. Sends initial code to Flask app's /execute endpoint
    3. Shuts down (execution, review, retries handled by async Flask app)
    
    Args:
        event: Lambda event - can be:
            - Direct invocation: {'paper_id': '...', 'slack_thread_ts': '...'}
            - SQS event: {'Records': [{'body': '{"paper_id": "..."}'}]}
        context: Lambda context
        
    Returns:
        Response dictionary with pipeline results (or batch results for SQS)
    """
    try:
        logger.info("Pipeline Lambda invoked")
        logger.info(f"Event type: {'SQS' if 'Records' in event else 'Direct'}")
        
        handler = PipelineHandler()
        
        # Check if this is an SQS event
        if 'Records' in event:
            # Parse all records first
            papers_to_process = []
            record_map = {}  # Map paper_id to record for error handling
            
            for i, record in enumerate(event['Records']):
                try:
                    # Parse SQS message body
                    if 'body' in record:
                        body = json.loads(record['body'])
                        paper_id = body.get('paper_id')
                        slack_thread_ts = body.get('slack_thread_ts')  # Optional
                    else:
                        # Fallback: try to get paper_id directly from record
                        paper_id = record.get('paper_id')
                        slack_thread_ts = record.get('slack_thread_ts')
                    
                    if not paper_id:
                        logger.warning(f"Record {i} missing paper_id, skipping")
                        continue
                    
                    papers_to_process.append({
                        'paper_id': paper_id,
                        'slack_thread_ts': slack_thread_ts,
                        'record_index': i,
                        'message_id': record.get('messageId', str(i))
                    })
                    record_map[paper_id] = record
                    
                except Exception as e:
                    logger.error(f"Error parsing record {i}: {e}")
            
            if not papers_to_process:
                return {
                    "batchItemFailures": [],
                    "results": [],
                    "processed": 0,
                    "failed": 0
                }
            
            logger.info(f"Processing {len(papers_to_process)} papers in parallel (max 5 concurrent)")
            
            # Process papers in parallel (max 5 concurrent to respect Bedrock limits)
            MAX_PARALLEL = 5
            results = []
            batch_item_failures = []
            results_map = {}  # Map paper_id to result
            
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
                # Submit all papers for processing
                future_to_paper = {
                    executor.submit(handler.process_paper, paper['paper_id'], paper.get('slack_thread_ts')): paper
                    for paper in papers_to_process
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    paper_id = paper['paper_id']
                    try:
                        result = future.result()
                        results_map[paper_id] = result
                        results.append(result)
                        
                        if not result.get('success'):
                            batch_item_failures.append({"itemIdentifier": paper['message_id']})
                            logger.warning(f"Paper {paper_id} processing failed")
                        else:
                            logger.info(f"✅ Paper {paper_id} processed successfully")
                            
                    except Exception as e:
                        logger.error(f"Error processing paper {paper_id}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        batch_item_failures.append({"itemIdentifier": paper['message_id']})
                        results.append({
                            "paper_id": paper_id,
                            "success": False,
                            "error": str(e)
                        })
            
            # Collect papers with generated code for Trainium batching
            papers_with_code = []
            for paper in papers_to_process:
                paper_id = paper['paper_id']
                result = results_map.get(paper_id)
                if result and result.get('success') and result.get('code'):
                    papers_with_code.append({
                        'paper_id': paper_id,
                        'code': result.get('code'),
                        'paper_title': result.get('paper_title'),
                        'slack_thread_ts': result.get('slack_thread_ts') or paper.get('slack_thread_ts')  # Use from result first, fallback to original
                    })
            
            # Batch send to Trainium (up to 10 papers at once, sent concurrently)
            trainium_result = None
            if papers_with_code:
                logger.info(f"Sending {len(papers_with_code)} papers to Trainium in batches of 10")
                trainium_result = send_papers_to_trainium_batch(papers_with_code, batch_size=10)
            
            # Return batch response for SQS
            return {
                "batchItemFailures": batch_item_failures,
                "results": results,
                "processed": len(results),
                "failed": len(batch_item_failures),
                "trainium_sent": trainium_result.get('sent', 0) if trainium_result else 0,
                "trainium_failed": trainium_result.get('failed', 0) if trainium_result else 0
            }
        else:
            # Direct invocation (non-SQS)
            paper_id = event.get('paper_id')
            if not paper_id:
                return {
                    "success": False,
                    "error": "paper_id is required in event"
                }
            
            slack_thread_ts = event.get('slack_thread_ts')  # Optional
            
            # Process paper: generate code (will be sent to Trainium in batch if part of batch)
            result = handler.process_paper(paper_id, slack_thread_ts=slack_thread_ts)
            
            # For direct invocation, send to Trainium immediately (not batched)
            if result.get('success') and result.get('code'):
                flask_result = send_to_flask_app(
                    paper_id=paper_id,
                    code=result['code'],
                    paper_title=result.get('paper_title'),
                    slack_thread_ts=slack_thread_ts
                )
                result["flask_app"] = flask_result
                result["success"] = flask_result.get("success", False)
            
            return result
        
    except Exception as e:
        logger.error(f"Error in Pipeline Lambda: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }
