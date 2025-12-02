"""
Lambda handler for PapersCodeGenerator.

This is the unified entry point for both Lambda and CLI usage.
Handles code generation, S3 storage, OpenSearch updates, and SQS queueing.
"""

import os
import json
import boto3
import logging
import argparse
import time
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import sys

sys.path.append('/opt/python')  
sys.path.append(os.path.dirname(__file__))

chunked_path = os.path.join(os.path.dirname(__file__), 'chunked-code-gen')
if os.path.exists(chunked_path) and chunked_path not in sys.path:
    sys.path.insert(0, chunked_path)

# Importing chunked generator to handle diff enviornments (Docker, local, etc)
try:
    from chunked_generator import ChunkedPyTorchGenerator  # type: ignore
except ImportError:
    import importlib.util
    chunked_gen_path = os.path.join(chunked_path, 'chunked_generator.py')
    if os.path.exists(chunked_gen_path):
        spec = importlib.util.spec_from_file_location("chunked_generator", chunked_gen_path)
        chunked_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunked_module)
        ChunkedPyTorchGenerator = chunked_module.ChunkedPyTorchGenerator
    else:
        raise ImportError("Could not import ChunkedPyTorchGenerator from chunked-code-gen")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

s3_client = None
sqs_client = None
os_client = None

CODE_BUCKET = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
CODE_TEST_QUEUE_URL = os.getenv('CODE_TEST_QUEUE_URL')  # code-testing.fifo queue
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'research-papers-v2')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

def init_aws_clients():
    """Initialize AWS clients (only in Lambda environment)"""
    global s3_client, sqs_client, os_client
    
    if s3_client is None:
        s3_client = boto3.client('s3')
        sqs_client = boto3.client('sqs')
        
        # Setup OpenSearch client
        if OPENSEARCH_ENDPOINT:
            session = boto3.Session()
            creds = session.get_credentials().get_frozen_credentials()
            auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
            
            os_client = OpenSearch(
                hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://","").replace("http://",""), "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=60
            )
            logger.info("AWS clients initialized")

def save_to_s3(paper_id: str, code: str, code_result: Dict[str, Any]) -> Dict[str, str]:
    """Save generated code and metadata to S3"""
    if s3_client is None:
        raise RuntimeError("S3 client not initialized")
    
    prefix = f"{paper_id}/"
    
    # Save code
    code_key = f"{prefix}code.py"
    s3_client.put_object(
        Bucket=CODE_BUCKET,
        Key=code_key,
        Body=code.encode('utf-8'),
        ContentType='text/x-python'
    )
    
    metadata = {
        "paper_id": code_result.get("paper_id"),
        "paper_title": code_result.get("paper_title"),
        "paper_authors": code_result.get("paper_authors", []),
        "explanation": code_result.get("explanation"),  # Contains key info about metrics/improvements
        "generated_at": code_result.get("generated_at"),
        "model_used": code_result.get("model_used"),
        "s3_code_location": f"s3://{CODE_BUCKET}/{code_key}",
        "recommended_dataset": code_result.get("recommended_dataset"),
        "dataset_recommendations": code_result.get("dataset_recommendations"),
        "code_review": code_result.get("code_review")  # Code review metadata (fixes_applied, iterations, etc.)
    }
    
    # Save metadata
    metadata_key = f"{prefix}metadata.json"
    s3_client.put_object(
        Bucket=CODE_BUCKET,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    
    logger.info(f"Saved code to s3://{CODE_BUCKET}/{code_key}")
    
    return {
        "s3_bucket": CODE_BUCKET,
        "code_key": code_key,
        "metadata_key": metadata_key
    }

def update_opensearch(paper_id: str, code_info: Dict[str, Any], dataset_info: Optional[Dict[str, Any]] = None):
    """Update OpenSearch document with code generation status"""
    if os_client is None:
        logger.warning("OpenSearch client not initialized, skipping update")
        return
    
    try:
        doc_update = {
            "code_generated": True,
            "code_s3_bucket": code_info["s3_bucket"],
            "code_s3_key": code_info["code_key"],
            "code_generated_at": datetime.now().isoformat(),
            "code_metadata_s3_key": code_info["metadata_key"]
        }
        
        # Add dataset information if available
        if dataset_info:
            doc_update["recommended_dataset"] = dataset_info.get("recommended_dataset")
            doc_update["dataset_recommendations"] = dataset_info.get("dataset_recommendations", {})
        
        os_client.update(
            index=OPENSEARCH_INDEX,
            id=paper_id,
            body={
                "doc": doc_update
            }
        )
        logger.info(f"Updated OpenSearch document {paper_id} with code generation status")
        
    except Exception as e:
        logger.error(f"Error updating OpenSearch: {e}")
        raise

def send_to_testing_queue(paper_id: str, paper_title: str, s3_info: Dict[str, str]):
    """Send message to code-testing.fifo for Trainium testing"""
    if not CODE_TEST_QUEUE_URL:
        logger.warning("CODE_TEST_QUEUE_URL not set, skipping SQS")
        return
    
    if sqs_client is None:
        raise RuntimeError("SQS client not initialized")
    
    message = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "s3_bucket": s3_info["s3_bucket"],
        "s3_code_key": s3_info["code_key"],
        "s3_metadata_key": s3_info["metadata_key"],
        "queued_at": datetime.now().isoformat()
    }
    
    sqs_client.send_message(
        QueueUrl=CODE_TEST_QUEUE_URL,
        MessageBody=json.dumps(message),
        MessageGroupId=paper_id,
        MessageDeduplicationId=f"{paper_id}-{datetime.now().isoformat()}"
    )
    
    logger.info(f"Sent to code-testing queue for paper {paper_id}")


class CodeGenHandler:
    """Main handler for the code generation system."""
    
    def __init__(self):
        """Initialize the code generation handler."""
        # This processes papers in PDF chunks and prioritizes relevant sections (filters out appendix, etc.)
        self.generator = ChunkedPyTorchGenerator(
            batch_size=8,  # Group 8 chunk summaries into each batch for hierarchical summarization
            use_smart_pdf_chunking=True,  # Enable smart chunking to prioritize relevant sections (filters appendix)
            max_pdf_chunks=15,  # Maximum number of PDF chunks to process (prioritizes abstract, formulas, diagrams)
            pages_per_pdf_chunk=2  # 2 pages per PDF chunk
        )
        logger.info("Code Gen Handler initialized with ChunkedPyTorchGenerator (PDF chunking enabled)")
    
    def handle_lambda_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle AWS Lambda events for code generation.
        Always uses AWS integrations (S3, OpenSearch, SQS).
        
        Args:
            event: Lambda event dictionary
            
        Returns:
            Response dictionary
        """
        try:
            # Extract parameters from event
            action = event.get('action', 'generate_by_id')
            paper_id = event.get('paper_id')
            paper_ids = event.get('paper_ids', [])
            title = event.get('title')
            author = event.get('author')
            keywords = event.get('keywords')
            max_papers = event.get('max_papers', 5)
            include_full_content = event.get('include_full_content', True)
            days = event.get('days', 30)
            
            logger.info(f"Processing action: {action}")
            
            # Route to appropriate handler
            if action == 'generate_by_id':
                if not paper_id:
                    return {"error": "paper_id is required for generate_by_id action"}
                # ChunkedPyTorchGenerator doesn't take include_full_content param - it always processes full content
                result = self.generator.generate_code_for_paper(paper_id)
                
            elif action == 'generate_by_ids':
                if not paper_ids:
                    return {"error": "paper_ids is required for generate_by_ids action"}
                # ChunkedPyTorchGenerator always processes full content (include_full_content param ignored)
                result = self.generator.generate_code_for_papers(paper_ids, include_full_content)
                
            elif action == 'generate_by_title':
                if not title:
                    return {"error": "title is required for generate_by_title action"}
                result = self.generator.generate_code_by_title(title, max_papers, include_full_content)
                
            elif action == 'generate_by_author':
                if not author:
                    return {"error": "author is required for generate_by_author action"}
                result = self.generator.generate_code_by_author(author, max_papers, include_full_content)
                
            elif action == 'generate_by_keywords':
                if not keywords:
                    return {"error": "keywords is required for generate_by_keywords action"}
                result = self.generator.generate_code_by_keywords(keywords, max_papers, include_full_content)
                
            elif action == 'generate_recent':
                result = self.generator.generate_code_for_recent_papers(days, max_papers, include_full_content)
                
            elif action == 'get_paper_info':
                if not paper_id:
                    return {"error": "paper_id is required for get_paper_info action"}
                result = self.generator.get_paper_info(paper_id)
                
            else:
                return {"error": f"Unknown action: {action}"}
            
            # Add metadata to response
            result.update({
                "action": action,
                "timestamp": result.get("generated_at", "unknown")
            })
            
            # Handle AWS integrations if enabled and successful
            # Only proceed if code generation AND code review succeeded
            if result.get('success') and result.get('code') and action != 'get_paper_info':
                result = self._handle_aws_integrations(result)
            elif not result.get('success'):
                # Code generation failed - log and continue (don't send to test queue)
                logger.error(f"Code generation failed for paper {result.get('paper_id')}: {result.get('error')}")
                # Don't update OpenSearch or send to queue - just dump this paper
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling lambda event: {e}")
            return {
                "error": f"Error: {str(e)}",
                "action": event.get('action', 'unknown')
            }
    
    def _handle_aws_integrations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AWS integrations (S3, OpenSearch, SQS) for successful code generation."""
        try:
            # Extract result (handle both single and multiple results)
            if 'results' in result:
                code_result = result['results'][0]
            else:
                code_result = result
            
            if not code_result.get('success'):
                return result
            
            if code_result.get('already_generated') is True:
                logger.info(f"Skipping AWS integrations: code already generated for paper {code_result.get('paper_id')}")
                return result
            
            # Extract details
            paper_id = code_result.get('paper_id')
            paper_title = code_result.get('paper_title')
            code = code_result.get('code')
            
            # Save to S3
            s3_info = save_to_s3(paper_id, code, code_result)
            
            # Prepare dataset info for OpenSearch update
            dataset_info = {
                "recommended_dataset": code_result.get("recommended_dataset"),
                "dataset_recommendations": code_result.get("dataset_recommendations")
            }
            
            # Update OpenSearch
            update_opensearch(paper_id, s3_info, dataset_info)
            
            # Send to code-testing queue for Trainium testing
            send_to_testing_queue(paper_id, paper_title, s3_info)
            
            # Add info to response
            code_result['s3_bucket'] = s3_info['s3_bucket']
            code_result['s3_code_key'] = s3_info['code_key']
            code_result['queued_for_testing'] = True
            code_result['opensearch_updated'] = True
            
            return code_result
            
        except Exception as e:
            logger.error(f"Error in AWS integrations: {e}")
            result['aws_integration_error'] = str(e)
            return result
    
    def run_cli(self, args: List[str] = None) -> None:
        """
        Run the code generator from command line.
        
        Args:
            args: Command line arguments
        """
        parser = argparse.ArgumentParser(description='Generate PyTorch code from research papers')
        
        # Action selection
        parser.add_argument('action', choices=[
            'generate_by_id', 'generate_by_ids', 'generate_by_title', 
            'generate_by_author', 'generate_by_keywords', 'generate_recent', 'get_paper_info'
        ], help='Action to perform')
        
        # Parameters
        parser.add_argument('--paper-id', help='Paper ID for single paper operations')
        parser.add_argument('--paper-ids', nargs='+', help='List of paper IDs')
        parser.add_argument('--title', help='Paper title to search for')
        parser.add_argument('--author', help='Author name to search for')
        parser.add_argument('--keywords', help='Keywords to search in abstract')
        parser.add_argument('--max-papers', type=int, default=5, help='Maximum number of papers to process')
        parser.add_argument('--include-full-content', action='store_true', help='Include full paper content')
        parser.add_argument('--days', type=int, default=30, help='Days to look back for recent papers')
        parser.add_argument('--output-dir', default='generated_code', help='Output directory for generated code')
        parser.add_argument('--save', action='store_true', help='Save generated code to files')
        
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        try:
            # Convert to event format
            event = {
                'action': parsed_args.action,
                'paper_id': parsed_args.paper_id,
                'paper_ids': parsed_args.paper_ids,
                'title': parsed_args.title,
                'author': parsed_args.author,
                'keywords': parsed_args.keywords,
                'max_papers': parsed_args.max_papers,
                'include_full_content': parsed_args.include_full_content,
                'days': parsed_args.days
            }
            
            # Process the request (CLI mode - still uses AWS for S3/OpenSearch)
            result = self.handle_lambda_event(event)
            
            # Print results
            print(json.dumps(result, indent=2))
            
            # Save code if requested
            if parsed_args.save and result.get('success'):
                if 'results' in result:
                    # Multiple results
                    for res in result['results']:
                        if res.get('success'):
                            self.generator.save_generated_code(res, parsed_args.output_dir)
                else:
                    # Single result
                    self.generator.save_generated_code(result, parsed_args.output_dir)
            
        except Exception as e:
            logger.error(f"Error in CLI: {e}")
            print(f"Error: {e}")


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    Handles both direct invocations and SQS batch events.
    Generates code, saves to S3, updates OpenSearch, and queues for testing.
    
    Args:
        event: Lambda event (direct or SQS batch)
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    try:
        logger.info("PapersCodeGenerator Lambda invoked")
        
        # Initialize AWS clients
        init_aws_clients()
        
        handler = CodeGenHandler()
        
        # Check if this is an SQS batch event
        if 'Records' in event:
            logger.info(f"Processing SQS batch with {len(event['Records'])} messages")
            
            results = []
            failures = []
            
            # Get remaining time to avoid timeout (leave 30 seconds buffer)
            remaining_time_ms = context.get_remaining_time_in_millis() if context else None
            timeout_buffer_ms = 30000  # 30 seconds buffer
            
            # Per-paper timeout: 3 minutes (180 seconds) max per paper
            # This prevents one stuck paper from blocking the entire batch
            PER_PAPER_TIMEOUT_SECONDS = 180  # 3 minutes per paper
            
            for record in event['Records']:
                # Check if we're running out of Lambda time
                if remaining_time_ms and remaining_time_ms < timeout_buffer_ms:
                    logger.warning(f"Lambda timeout approaching ({remaining_time_ms}ms remaining). Marking remaining {len(event['Records']) - len(results)} messages for retry.")
                    # Mark all remaining messages for retry
                    for remaining_record in event['Records'][len(results):]:
                        failures.append({
                            "itemIdentifier": remaining_record.get('messageId')
                        })
                    break
                
                try:
                    # Parse message body
                    message = json.loads(record['body'])
                    paper_id = message.get('paper_id')
                    
                    # Check if code already generated (skip if done to save time)
                    try:
                        if os_client:
                            doc = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
                            if doc['_source'].get('code_generated', False):
                                logger.info(f"Skipping {paper_id} - code already generated")
                                continue
                    except Exception as check_error:
                        # If check fails, proceed with generation
                        logger.debug(f"Could not check code_generated status: {check_error}")
                    
                    logger.info(f"Generating code for paper: {paper_id} (timeout: {PER_PAPER_TIMEOUT_SECONDS}s)")
                    
                    # Track start time for per-paper timeout
                    paper_start_time = time.time()
                    
                    # Generate code for this paper with per-paper timeout
                    # Use ThreadPoolExecutor with timeout to interrupt blocking I/O calls
                    try:
                        class PaperTimeoutError(Exception):
                            """Custom timeout exception for per-paper timeouts"""
                            pass
                        
                        def generate_code_with_timeout():
                            """Wrapper function to run code generation in a thread"""
                            return handler.handle_lambda_event(message)
                        
                        # Use ThreadPoolExecutor with timeout to interrupt blocking calls
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(generate_code_with_timeout)
                            try:
                                result = future.result(timeout=PER_PAPER_TIMEOUT_SECONDS)
                                results.append(result)
                                paper_elapsed = time.time() - paper_start_time
                                logger.info(f"Paper {paper_id} completed in {paper_elapsed:.1f}s")
                            except FuturesTimeoutError:
                                # Cancel the future (though it may continue running)
                                future.cancel()
                                raise PaperTimeoutError(f"Paper {paper_id} exceeded {PER_PAPER_TIMEOUT_SECONDS}s timeout")
                        
                    except PaperTimeoutError as timeout_err:
                        paper_elapsed = time.time() - paper_start_time
                        logger.error(f"Paper {paper_id} timed out after {paper_elapsed:.1f}s: {timeout_err}")
                        # Mark this paper as failed and continue with next paper (dump it)
                        failures.append({
                            "itemIdentifier": record.get('messageId')
                        })
                        # Update OpenSearch to record the timeout
                        try:
                            if os_client:
                                os_client.update(
                                    index=OPENSEARCH_INDEX,
                                    id=paper_id,
                                    body={
                                        "doc": {
                                            "code_generation_error": f"Timeout after {PER_PAPER_TIMEOUT_SECONDS}s",
                                            "code_generation_failed_at": datetime.now().isoformat()
                                        }
                                    }
                                )
                        except Exception as update_err:
                            logger.warning(f"Failed to update OpenSearch with timeout error: {update_err}")
                        continue  # Move to next paper (dump this one)
                    
                    # Check if code generation succeeded
                    if not result.get('success') or not result.get('code'):
                        logger.error(f"Code generation failed for paper {paper_id}: {result.get('error', 'Unknown error')}")
                        # Dump this paper - mark as failed and continue
                        failures.append({
                            "itemIdentifier": record.get('messageId')
                        })
                        # Update OpenSearch to record the failure
                        try:
                            if os_client:
                                os_client.update(
                                    index=OPENSEARCH_INDEX,
                                    id=paper_id,
                                    body={
                                        "doc": {
                                            "code_generation_error": result.get('error', 'Code generation failed'),
                                            "code_generation_failed_at": datetime.now().isoformat()
                                        }
                                    }
                                )
                        except Exception as update_err:
                            logger.warning(f"Failed to update OpenSearch with error: {update_err}")
                        continue  # Move to next paper (dump this one)
                    
                    # Update remaining time after processing
                    if remaining_time_ms:
                        remaining_time_ms = context.get_remaining_time_in_millis() if context else None
                    
                except Exception as e:
                    logger.error(f"Error processing record: {e}")
                    # Dump this paper on any exception
                    failures.append({
                        "itemIdentifier": record.get('messageId')
                    })
            
            return {
                "success": True,
                "processed": len(results),
                "failed": len(failures),
                "batchItemFailures": failures  # SQS partial batch failure handling
            }
        
        else:
            # Direct invocation (not from SQS)
            result = handler.handle_lambda_event(event)
            return result
        
    except Exception as e:
        logger.error(f"Error in PapersCodeGenerator Lambda: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def main():
    """Main entry point for CLI usage."""
    handler = CodeGenHandler()
    handler.run_cli()


if __name__ == "__main__":
    main()
