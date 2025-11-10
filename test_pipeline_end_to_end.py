#!/usr/bin/env python3
"""
End-to-end pipeline test script.

This script tests the full queue-based pipeline:
1. Resets code_generated flag for specified papers
2. Sends papers to code-evaluation queue (triggers code generation)
3. Waits for code generation to complete (code gen Lambda auto-sends to code-testing queue)
4. Waits for Trainium execution to complete (Trainium Lambda processes code-testing queue)
5. Reports results from OpenSearch
6. Provides detailed logging throughout

The pipeline flow:
  code-evaluation.fifo → Code Gen Lambda → code-testing.fifo → Trainium Lambda → Trainium

Usage:
    python test_pipeline_end_to_end.py
"""

import os
import sys
import json
import time
import logging
import boto3
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add code_gen to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code_gen'))

load_dotenv()

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v2")
CODE_EVAL_QUEUE_NAME = "code-evaluation.fifo"

# Papers to test (4 papers for testing - reduced from 8 to avoid Lambda timeout)
PAPERS = [
    ("XOhHZpoBclM7MZc3dZMP", "MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents"),
    ("j-hHZpoBclM7MZc3vpPD", "LoCA: Location-Aware Cosine Adaptation for Parameter-Efficient Fine-Tuning"),
    ("9OhIZpoBclM7MZc3RpN3", "ADAM Optimization with Adaptive Batch Selection"),
    ("-OhTW5oBclM7MZc3T5IM", "DelTA: An Online Document-Level Translation Agent Based on Multi-Level Memory"),
]

def reset_paper_code_generated(paper_id: str) -> bool:
    """Reset code_generated flag for a paper using reset_code_generated.py"""
    logger.info(f"🔄 Resetting code_generated for paper: {paper_id}")
    try:
        result = subprocess.run(
            [sys.executable, "reset_code_generated.py", paper_id],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info(f"✅ Successfully reset code_generated for {paper_id}")
            return True
        else:
            logger.error(f"❌ Failed to reset {paper_id}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error resetting {paper_id}: {e}")
        return False

def get_code_eval_queue_url() -> Optional[str]:
    """Get the code-evaluation queue URL"""
    try:
        sqs = boto3.client('sqs', region_name=AWS_REGION)
        response = sqs.get_queue_url(QueueName=CODE_EVAL_QUEUE_NAME)
        return response['QueueUrl']
    except Exception as e:
        logger.error(f"❌ Failed to get queue URL: {e}")
        return None

def send_paper_to_code_eval_queue(paper_id: str, paper_title: str, queue_url: str) -> bool:
    """Send a paper to the code-evaluation queue"""
    logger.info(f"📤 Sending paper to code-evaluation queue: {paper_title[:60]}...")
    try:
        sqs = boto3.client('sqs', region_name=AWS_REGION)
        message = {
            "paper_id": paper_id,
            "action": "generate_by_id",
            "paper_title": paper_title,
            "queued_at": datetime.now().isoformat()
        }
        
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
            MessageGroupId=paper_id,
            MessageDeduplicationId=f"{paper_id}-{int(time.time() * 1000)}"
        )
        
        logger.info(f"✅ Sent paper {paper_id} to queue (MessageId: {response['MessageId']})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send {paper_id} to queue: {e}")
        return False

def get_opensearch_client():
    """Get OpenSearch client"""
    from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
    
    session = boto3.Session(region_name=AWS_REGION)
    credentials = session.get_credentials().get_frozen_credentials()
    auth = AWSV4SignerAuth(credentials, AWS_REGION, "es")
    
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", ""), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )

def wait_for_code_generation(paper_id: str, max_wait_minutes: int = 10) -> bool:
    """Wait for code generation to complete by checking OpenSearch"""
    logger.info(f"⏳ Waiting for code generation for paper: {paper_id}")
    
    try:
        os_client = get_opensearch_client()
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval = 10  # Check every 10 seconds
        
        while time.time() - start_time < max_wait_seconds:
            try:
                doc = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
                source = doc['_source']
                code_generated = source.get('code_generated', False)
                code_generation_error = source.get('code_generation_error')
                
                if code_generated:
                    logger.info(f"✅ Code generation complete for {paper_id}")
                    logger.info(f"   Code will be automatically queued to code-testing.fifo")
                    return True
                
                # Check if paper timed out or failed
                if code_generation_error:
                    logger.error(f"❌ Code generation failed for {paper_id}: {code_generation_error}")
                    return False
                
                elapsed = int(time.time() - start_time)
                logger.info(f"⏳ Still waiting... ({elapsed}s elapsed)")
                time.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking status: {e}, retrying...")
                time.sleep(check_interval)
        
        logger.error(f"❌ Timeout waiting for code generation for {paper_id}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error waiting for code generation: {e}")
        return False

def wait_for_trainium_execution(paper_id: str, max_wait_minutes: int = 60) -> Dict[str, Any]:
    """
    Wait for Trainium execution to complete by checking OpenSearch.
    Returns execution status dictionary.
    """
    logger.info(f"⏳ Waiting for Trainium execution for paper: {paper_id}")
    
    try:
        os_client = get_opensearch_client()
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval = 15  # Check every 15 seconds
        
        while time.time() - start_time < max_wait_seconds:
            try:
                doc = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
                source = doc['_source']
                tested = source.get('tested', False)
                
                if tested:
                    test_success = source.get('test_success', False)
                    execution_time = source.get('execution_time')
                    return_code = source.get('return_code')
                    error_message = source.get('error_message')
                    
                    logger.info(f"✅ Trainium execution complete for {paper_id}")
                    logger.info(f"   Success: {test_success}, Time: {execution_time}s, Return code: {return_code}")
                    if error_message:
                        logger.warning(f"   Error: {error_message}")
                    
                    return {
                        'tested': True,
                        'test_success': test_success,
                        'execution_time': execution_time,
                        'return_code': return_code,
                        'error_message': error_message,
                        'tested_at': source.get('tested_at')
                    }
                
                elapsed = int(time.time() - start_time)
                logger.info(f"⏳ Still waiting for Trainium execution... ({elapsed}s elapsed)")
                time.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"⚠️ Error checking status: {e}, retrying...")
                time.sleep(check_interval)
        
        logger.error(f"❌ Timeout waiting for Trainium execution for {paper_id}")
        return {
            'tested': False,
            'test_success': False,
            'error_message': 'Timeout waiting for execution'
        }
        
    except Exception as e:
        logger.error(f"❌ Error waiting for Trainium execution: {e}")
        return {
            'tested': False,
            'test_success': False,
            'error_message': str(e)
        }

def main():
    """Main pipeline test function"""
    logger.info("=" * 80)
    logger.info("PIPELINE END-TO-END TEST")
    logger.info("=" * 80)
    logger.info(f"Testing {len(PAPERS)} papers")
    logger.info("")
    
    # Step 1: Get queue URL
    logger.info("STEP 1: Getting code-evaluation queue URL...")
    queue_url = get_code_eval_queue_url()
    if not queue_url:
        logger.error("❌ Failed to get queue URL. Exiting.")
        return 1
    logger.info(f"✅ Queue URL: {queue_url}")
    logger.info("")
    
    # Step 2: Reset code_generated flags
    logger.info("STEP 2: Resetting code_generated flags...")
    reset_results = {}
    for paper_id, paper_title in PAPERS:
        reset_results[paper_id] = reset_paper_code_generated(paper_id)
    logger.info(f"✅ Reset {sum(reset_results.values())}/{len(PAPERS)} papers")
    logger.info("")
    
    # Step 3: Send papers to queue
    logger.info("STEP 3: Sending papers to code-evaluation queue...")
    logger.info("⚠️  Note: Lambda timeout is 15 min max. Processing 4 papers sequentially may take 8-12 min.")
    logger.info("⚠️  Papers will be processed in batches by SQS (batch size: 10, immediate processing)")
    logger.info("⚠️  If Lambda times out, SQS will retry remaining papers automatically")
    queue_results = {}
    for paper_id, paper_title in PAPERS:
        queue_results[paper_id] = send_paper_to_code_eval_queue(paper_id, paper_title, queue_url)
    logger.info(f"✅ Queued {sum(queue_results.values())}/{len(PAPERS)} papers")
    logger.info("")
    
    # Step 4: Wait for code generation
    logger.info("STEP 4: Waiting for code generation...")
    logger.info("⚠️ This may take 20-30 minutes total (4 Bedrock calls per paper, ~2-3 min per paper)...")
    logger.info("⚠️ Lambda processes papers in batches. If a batch times out, remaining papers will be retried.")
    logger.info("⚠️ Code generation Lambda will automatically send papers to code-testing queue when done.")
    generation_results = {}
    for paper_id, paper_title in PAPERS:
        if queue_results.get(paper_id):
            # Increase wait time since Lambda may need to process in multiple batches
            generation_results[paper_id] = wait_for_code_generation(paper_id, max_wait_minutes=30)
        else:
            logger.warning(f"⚠️ Skipping {paper_id} (not queued)")
            generation_results[paper_id] = False
    logger.info(f"✅ Code generated for {sum(generation_results.values())}/{len(PAPERS)} papers")
    logger.info("")
    
    # Step 5: Wait for Trainium execution
    logger.info("STEP 5: Waiting for Trainium execution...")
    logger.info("⚠️ Papers are automatically queued to code-testing.fifo after code generation")
    logger.info("⚠️ Trainium Lambda processes batches of 10 (immediate processing)")
    logger.info("⚠️ This may take 30-60 minutes total depending on queue depth and Trainium startup...")
    execution_results = {}
    for paper_id, paper_title in PAPERS:
        if generation_results.get(paper_id):
            exec_status = wait_for_trainium_execution(paper_id, max_wait_minutes=60)
            execution_results[paper_id] = exec_status
        else:
            logger.warning(f"⚠️ Skipping {paper_id} (code not generated)")
            execution_results[paper_id] = {'tested': False, 'test_success': False}
    
    successful_executions = sum(1 for r in execution_results.values() if r.get('test_success', False))
    logger.info(f"✅ Trainium execution complete for {successful_executions}/{sum(generation_results.values())} papers")
    logger.info("")
    
    # Final summary
    logger.info("=" * 80)
    logger.info("PIPELINE TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Papers tested: {len(PAPERS)}")
    logger.info(f"  ✅ Reset: {sum(reset_results.values())}")
    logger.info(f"  ✅ Queued: {sum(queue_results.values())}")
    logger.info(f"  ✅ Code generated: {sum(generation_results.values())}")
    successful_executions = sum(1 for r in execution_results.values() if r.get('test_success', False))
    tested_count = sum(1 for r in execution_results.values() if r.get('tested', False))
    logger.info(f"  ✅ Tested on Trainium: {tested_count}/{sum(generation_results.values())}")
    logger.info(f"  ✅ Successful executions: {successful_executions}/{tested_count}")
    logger.info("")
    
    # Detailed results
    logger.info("Detailed Results:")
    for paper_id, paper_title in PAPERS:
        status = []
        if reset_results.get(paper_id): status.append("Reset")
        if queue_results.get(paper_id): status.append("Queued")
        if generation_results.get(paper_id): status.append("Generated")
        
        exec_status = execution_results.get(paper_id, {})
        if exec_status.get('tested'):
            if exec_status.get('test_success'):
                status.append("✅ Executed")
            else:
                status.append("❌ Failed")
                if exec_status.get('error_message'):
                    status.append(f"({exec_status['error_message'][:30]}...)")
        elif generation_results.get(paper_id):
            status.append("⏳ Waiting")
        
        logger.info(f"  {paper_id[:16]}... | {paper_title[:50]:50} | {' → '.join(status)}")
    
    logger.info("")
    logger.info("✅ Pipeline test complete! Check OpenSearch for detailed execution results.")
    logger.info("   Use check_trainium_execution.py <paper_id> to view detailed results for a paper.")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

