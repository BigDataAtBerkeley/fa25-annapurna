"""
Lambda handler for Pipeline Code Generation.

This Lambda function:
1. Generates code using ChunkedPyTorchGenerator 
2. Enqueues generated code to the Trainium execution queue

All execution is handled asynchronously via the queue.
"""

import os
import json
import logging
import time
import boto3
from datetime import datetime
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError
import sys
import fitz

current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
library_paths = [
    '/lib64',
    '/usr/lib64', 
    '/var/lang/lib64',
    '/usr/lib',
    '/lib',
    '/var/task', 
]
os.environ['LD_LIBRARY_PATH'] = ':'.join(library_paths) + (':' + current_ld_path if current_ld_path else '')
if '/opt/python' not in sys.path:
    sys.path.insert(0, '/opt/python')
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from chunked_generator import ChunkedPyTorchGenerator  # type: ignore
from slack_notifier import SlackNotifier

TRAINIUM_EXECUTION_QUEUE_URL = os.getenv('TRAINIUM_EXECUTION_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo')

def enqueue_paper_for_execution(paper_id: str, code: str, paper_title: Optional[str] = None, 
                                slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
    """
    Enqueues a paper to the Trainium execution queue
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
            MessageGroupId=paper_id,  
            MessageDeduplicationId=f"{paper_id}-{int(time.time())}"  
        )
        logger.info(f"Enqueued paper {paper_id} to execution queue")
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
                result = enqueue_paper_for_execution(
                    paper['paper_id'],
                    paper['code'],
                    paper.get('paper_title'),
                    paper.get('slack_thread_ts')
                )
                
                if result.get('success'):
                    total_sent += 1
                    logger.info(f"Enqueued {paper['paper_id']} to Trainium")
                else:
                    total_failed += 1
                    logger.error(f"Failed to enqueue {paper['paper_id']}: {result.get('error')}")
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


class PipelineHandler:
    """Main handler for the pipeline Lambda (code generation only)."""
    
    def __init__(self):
        """Initialize the pipeline handler."""
        self.generator = ChunkedPyTorchGenerator(
            batch_size=8,  # Group 8 chunk summaries into each batch for hierarchical summarization
            max_pdf_chunks=15,  # Maximum number of PDF chunks to process (prioritizes abstract, formulas, diagrams)
            pages_per_pdf_chunk=2  # 2 pages per PDF chunk
        )

        logger.info("Pipeline Handler initialized (code generation only)")
    
    def process_paper(self, paper_id: str, slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single paper: generate code and enqueue it for execution
        """
        start_time = time.time()
        result = {
            "paper_id": paper_id,
            "success": False,
            "pipeline_start": datetime.now().isoformat()
        }
        
        try:
            # Step 0: Send initial paper notification to Slack (creates thread)
            if not slack_thread_ts:
                try:
                    paper = self.generator.opensearch_client.get_paper_by_id(paper_id)
                    if paper:
                        fields_to_keep = {'_id', 'title', 'authors', 'abstract', 's3_bucket', 's3_key', 'date', 'url', 'arxiv_id'}
                        filtered_paper = {k: v for k, v in paper.items() if k in fields_to_keep}
                        filtered_paper['_id'] = paper_id
                        slack_notifier = SlackNotifier()
                        slack_thread_ts = slack_notifier.send_paper_info(filtered_paper)
                        if slack_thread_ts:
                            logger.info(f"Sent initial paper notification to Slack")
                            # Store slack_thread_ts in OpenSearch for thread continuity
                            self.generator.opensearch_client.client.update(
                                index=self.generator.opensearch_client.opensearch_index,
                                id=paper_id,
                                body={
                                    "doc": {
                                        "slack_thread_ts": slack_thread_ts
                                    }
                                }
                            )
                        else:
                            logger.warning("Failed to send initial paper notification to Slack")
                    else:
                        logger.warning(f"Paper {paper_id} not found in OpenSearch - skipping initial Slack notification")
                except Exception as e:
                    logger.warning(f"Error sending initial paper notification to Slack: {e}")
            
            # Step 1: Generate code
            logger.info(f"Step 1: Generating code for paper {paper_id}...")
            code_gen_result = self.generator.generate_code_for_paper(paper_id)
            
            if not code_gen_result.get("success") or not code_gen_result.get("code"):
                result["error"] = code_gen_result.get("error", "Code generation failed")
                result["code_generation"] = {
                    "success": False,
                    "error": code_gen_result.get("error")
                }
                return result
            
            logger.info(f"Code generated ({len(code_gen_result['code']):,} chars)")
            result["code_generation"] = {
                "success": True,
                "code_length": len(code_gen_result["code"]),
                "model_used": code_gen_result.get("model_used"),
                "recommended_dataset": code_gen_result.get("recommended_dataset")
            }

            # Step 1.5: Code Reviewer 0 
            initial_code = code_gen_result["code"]
            reviewed_code = initial_code
            
            try:
                from code_reviewer_0 import code_reviewer_0

                paper_summary = None
                if hasattr(self.generator, 'opensearch_client'):
                    paper = self.generator.opensearch_client.get_paper_by_id(paper_id)
                    if paper:
                        paper_summary = self.generator.opensearch_client.get_paper_summary(paper)
                
                logger.info(f"Code Reviewer 0: Reviewing code for {paper_id}")
                review_result = code_reviewer_0(initial_code, paper_id, paper_summary)
                
                if review_result and review_result.get("code"):
                    reviewed_code = review_result["code"]
                    fixes_summary = review_result.get("fixes_summary", [])
                    code_changed = review_result.get("code_changed", False)
                    
                    if code_changed:
                        logger.info(f"Code Reviewer 0: Fixed TRN compatibility issues")
                        logger.info(f"   Fixes: {', '.join(fixes_summary) if isinstance(fixes_summary, list) else fixes_summary}")
                    else:
                        logger.info(f"Code Reviewer 0: No changes needed - code is already TRN compatible")
                    
                    result["code_review_0"] = {
                        "success": True,
                        "code_changed": code_changed,
                        "fixes_summary": fixes_summary
                    }
                else:
                    logger.warning(f"Code Reviewer 0 failed: No code returned")
                    logger.warning(f"Using original code without TRN compatibility fixes")
                    result["code_review_0"] = {
                        "success": False,
                        "error": "Code review failed - no code returned",
                        "code_changed": False
                    }
            except ImportError as e:
                logger.warning(f"Code Reviewer 0 module not available: {e}")
                result["code_review_0"] = {
                    "success": False,
                    "error": f"Code Reviewer 0 module not available: {str(e)}",
                    "code_changed": False,
                    "skipped": True
                }
            except Exception as e:
                logger.warning(f"Code Reviewer 0 error: {e} - using original code")
                import traceback
                logger.debug(f"Code Reviewer 0 traceback: {traceback.format_exc()}")
                result["code_review_0"] = {
                    "success": False,
                    "error": str(e),
                    "code_changed": False
                }
            
            # Update code to use reviewed version
            code_gen_result["code"] = reviewed_code

            # Step 1.6: Send code generation notification to Slack (first follow-up)
            if slack_thread_ts:
                try:
                    slack_notifier = SlackNotifier()
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
                    logger.info(f"Sent code generation notification to Slack")
                except Exception as e:
                    logger.warning(f"Failed to send code generation notification: {e}")

            # Step 2: Store reviewed code for batch sending to Trainium (don't send immediately)
            result["code"] = code_gen_result["code"] 
            result["paper_title"] = code_gen_result.get("paper_title")
            result["success"] = True  # Code generation succeeded
            result["pipeline_end"] = datetime.now().isoformat()
            result["total_pipeline_time"] = time.time() - start_time
            
            logger.info(f"Code generation complete for paper {paper_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            result["error"] = str(e)
            result["pipeline_end"] = datetime.now().isoformat()
            result["total_pipeline_time"] = time.time() - start_time
            return result


def lambda_handler(event, context):

    try:
        logger.info("Pipeline Lambda invoked")
        
        handler = PipelineHandler()
        
        if 'Records' not in event:
            logger.error("Invalid event: expected SQS Records")
            return {
                "batchItemFailures": [],
                "results": [],
                "processed": 0,
                "failed": 0,
                "error": "Invalid event format: expected SQS Records"
            }
        
        papers_to_process = []
        record_map = {}  
        
        for i, record in enumerate(event['Records']):
            try:
                if 'body' in record:
                    body = json.loads(record['body'])
                    paper_id = body.get('paper_id')
                    slack_thread_ts = body.get('slack_thread_ts')
                else:
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
        
        lambda_timeout_seconds = context.get_remaining_time_in_millis() / 1000.0 if context else 900.0
        safety_buffer_seconds = 60.0
        
        results = []
        batch_item_failures = []
        results_map = {}
        
        for paper in papers_to_process:
            paper_id = paper['paper_id']
            paper_index = papers_to_process.index(paper) + 1

            if context:
                remaining_time = context.get_remaining_time_in_millis() / 1000.0
                if remaining_time < safety_buffer_seconds:
                    logger.warning(f"Lambda timeout approaching ({remaining_time:.1f}s remaining)")
                    for remaining_paper in papers_to_process[paper_index - 1:]:
                        batch_item_failures.append({"itemIdentifier": remaining_paper['message_id']})
                        logger.info(f"Marked paper {remaining_paper['paper_id']} for retry (timeout protection)")
                    break
            
            logger.info(f"Processing paper {paper_id} ({paper_index}/{len(papers_to_process)})")
            
            try:
                result = handler.process_paper(paper_id, paper.get('slack_thread_ts'))
                results_map[paper_id] = result
                results.append(result)
                
                if not result.get('success'):
                    batch_item_failures.append({"itemIdentifier": paper['message_id']})
                    logger.warning(f"Paper {paper_id} processing failed: {result.get('error', 'Unknown error')}")
                else:
                    logger.info(f"Paper {paper_id} processed successfully")
                    
                    if paper_index < len(papers_to_process):
                        delay = 2.0
                        logger.info(f"Waiting {delay}s before processing next paper")
                        time.sleep(delay)
                        
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
        
        papers_with_code = []
        for paper in papers_to_process:
            paper_id = paper['paper_id']
            result = results_map.get(paper_id)
            if result and result.get('success') and result.get('code'):
                papers_with_code.append({
                    'paper_id': paper_id,
                    'code': result.get('code'),
                    'paper_title': result.get('paper_title'),
                    'slack_thread_ts': result.get('slack_thread_ts') or paper.get('slack_thread_ts')
                })
        
        trainium_result = None
        if papers_with_code:
            logger.info(f"Sending {len(papers_with_code)} papers to Trainium in batches of 10")
            trainium_result = send_papers_to_trainium_batch(papers_with_code, batch_size=10)
            if trainium_result:
                logger.info(f"Trainium: {trainium_result.get('sent', 0)} sent, {trainium_result.get('failed', 0)} failed")

        processed_count = len([r for r in results if r.get('success')])
        logger.info(
            f"Lambda execution summary: "
            f"Processed {processed_count}/{len(papers_to_process)} papers successfully, "
            f"{len(batch_item_failures)} marked for retry"
        )
        
        return {
            "batchItemFailures": batch_item_failures,
            "results": results,
            "processed": processed_count,
            "total": len(papers_to_process),
            "failed": len(batch_item_failures),
            "trainium_sent": trainium_result.get('sent', 0) if trainium_result else 0,
            "trainium_failed": trainium_result.get('failed', 0) if trainium_result else 0
        }
        
    except Exception as e:
        logger.error(f"Error in Pipeline Lambda: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }
