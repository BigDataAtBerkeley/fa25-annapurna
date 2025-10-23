"""
Code Tester Lambda - Batch Processing with Trainium

This Lambda:
1. Receives batch of messages from SQS (up to 10 at a time)
2. Downloads code from S3 for each paper
3. Sends batch of code to Trainium instance for execution
4. Waits for results from Trainium
5. Updates OpenSearch with test results
6. Saves artifacts (logs, plots) to S3
"""

import json
import os
import boto3
import logging
import requests
import time
from typing import Dict, Any, List
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')
ec2_client = boto3.client('ec2')
session = boto3.Session()

# Environment variables
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'research-papers-v2')
OUTPUTS_BUCKET = os.getenv('OUTPUTS_BUCKET', 'papers-test-outputs')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Trainium instance configuration
TRAINIUM_ENDPOINT = os.getenv('TRAINIUM_ENDPOINT')  # e.g., http://10.0.1.50:8000
TRAINIUM_INSTANCE_ID = os.getenv('TRAINIUM_INSTANCE_ID')  # EC2 instance ID (optional, for auto-start)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
TRAINIUM_TIMEOUT = int(os.getenv('TRAINIUM_TIMEOUT', '600'))  # 10 minutes default

# OpenSearch client setup
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

# def ensure_trainium_running() -> bool:
#     """
#     Ensure Trainium instance is running (if TRAINIUM_INSTANCE_ID is set).
#     Returns True if instance is running, False otherwise.
#     """
#     if not TRAINIUM_INSTANCE_ID:
#         logger.info("No TRAINIUM_INSTANCE_ID set, assuming instance is already running")
#         return True
    
#     try:
#         response = ec2_client.describe_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
#         state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
#         if state == 'running':
#             logger.info(f"Trainium instance {TRAINIUM_INSTANCE_ID} is running")
#             return True
#         elif state == 'stopped':
#             logger.info(f"Starting Trainium instance {TRAINIUM_INSTANCE_ID}...")
#             ec2_client.start_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
#             # Wait for instance to be running
#             waiter = ec2_client.get_waiter('instance_running')
#             waiter.wait(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
#             # Additional wait for services to start
#             time.sleep(30)
#             logger.info(f"Trainium instance {TRAINIUM_INSTANCE_ID} is now running")
#             return True
#         else:
#             logger.warning(f"Trainium instance {TRAINIUM_INSTANCE_ID} is in state: {state}")
#             return False
            
#     except Exception as e:
#         logger.error(f"Error checking/starting Trainium instance: {e}")
#         return False

def download_code_from_s3(bucket: str, key: str) -> str:
    """Download code from S3"""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

def save_output_to_s3(paper_id: str, filename: str, content: str) -> str:
    """Save output to S3 and return the key"""
    key = f"{paper_id}/outputs/{filename}"
    s3_client.put_object(
        Bucket=OUTPUTS_BUCKET,
        Key=key,
        Body=content.encode('utf-8')
    )
    logger.info(f"Saved {filename} to s3://{OUTPUTS_BUCKET}/{key}")
    return key

def update_opensearch(paper_id: str, test_results: Dict[str, Any]):
    """Update OpenSearch document with test results"""
    try:
        filtered_results = {k: v for k, v in test_results.items() if v is not None}
        
        os_client.update(
            index=OPENSEARCH_INDEX,
            id=paper_id,
            body={
                "doc": filtered_results
            }
        )
        logger.info(f"Updated OpenSearch document {paper_id} with test results")
        
    except Exception as e:
        logger.error(f"Error updating OpenSearch: {e}")
        raise

def send_batch_to_trainium(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Send batch of code to Trainium instance for execution.
    
    Args:
        batch: List of dicts with 'paper_id', 'paper_title', 'code', 's3_code_key'
    
    Returns:
        Dict mapping paper_id to execution results
    """
    if not TRAINIUM_ENDPOINT:
        raise ValueError("TRAINIUM_ENDPOINT environment variable not set")
    
    # Ensure Trainium instance is running
    if not ensure_trainium_running():
        raise RuntimeError("Failed to start Trainium instance")
    
    payload = {
        "batch": batch,
        "timeout": TRAINIUM_TIMEOUT
    }
    
    try:
        logger.info(f"Sending batch of {len(batch)} code files to Trainium at {TRAINIUM_ENDPOINT}")
        
        response = requests.post(
            f"{TRAINIUM_ENDPOINT}/execute_batch",
            json=payload,
            timeout=TRAINIUM_TIMEOUT + 30  # Add buffer to timeout
        )
        
        response.raise_for_status()
        results = response.json()
        
        logger.info(f"Received results from Trainium for {len(results.get('results', []))} papers")
        return results
        
    except requests.exceptions.Timeout:
        logger.error(f"Trainium execution timed out after {TRAINIUM_TIMEOUT} seconds")
        # Return timeout results for all papers in batch
        return {
            "success": False,
            "error": "Batch execution timed out",
            "results": {
                item['paper_id']: {
                    "success": False,
                    "execution_time": TRAINIUM_TIMEOUT,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": "",
                    "timeout": True,
                    "error_message": f"Batch execution timed out after {TRAINIUM_TIMEOUT} seconds",
                    "error_type": "timeout"
                }
                for item in batch
            }
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Trainium: {e}")
        # Return error results for all papers in batch
        return {
            "success": False,
            "error": str(e),
            "results": {
                item['paper_id']: {
                    "success": False,
                    "execution_time": 0,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "timeout": False,
                    "error_message": f"Communication error with Trainium: {str(e)}",
                    "error_type": "trainium_error"
                }
                for item in batch
            }
        }

def process_batch_results(batch_data: List[Dict[str, Any]], trainium_results: Dict[str, Any]):
    """
    Process results from Trainium and update OpenSearch with comprehensive metrics.
    
    Args:
        batch_data: Original batch data with paper info
        trainium_results: Results from Trainium execution
    """
    results_dict = trainium_results.get('results', {})
    
    for item in batch_data:
        paper_id = item['paper_id']
        exec_result = results_dict.get(paper_id, {
            "success": False,
            "error_message": "No result returned from Trainium",
            "error_type": "missing_result"
        })
        
        try:
            # Save stdout and stderr to S3
            stdout_key = save_output_to_s3(paper_id, 'stdout.log', exec_result.get('stdout', ''))
            stderr_key = save_output_to_s3(paper_id, 'stderr.log', exec_result.get('stderr', ''))
            
            # Save plots if any (Trainium might return these)
            plots_keys = []
            if 'plots' in exec_result:
                for plot_name, plot_data in exec_result.get('plots', {}).items():
                    try:
                        plot_key = save_output_to_s3(paper_id, f'plots/{plot_name}', plot_data)
                        plots_keys.append(plot_key)
                    except Exception as plot_error:
                        logger.warning(f"Failed to save plot {plot_name}: {plot_error}")
            
            # Save detailed metrics as JSON
            if exec_result.get('detailed_metrics'):
                try:
                    metrics_json = json.dumps(exec_result['detailed_metrics'], indent=2)
                    save_output_to_s3(paper_id, 'metrics.json', metrics_json)
                except Exception as metrics_error:
                    logger.warning(f"Failed to save metrics.json: {metrics_error}")
            
            # Calculate cost (trn1.2xlarge is ~$1.34/hour)
            execution_hours = exec_result.get('execution_time', 0) / 3600
            estimated_cost = execution_hours * 1.34
            
            # Prepare comprehensive test results for OpenSearch
            test_results = {
                # === Core Test Status ===
                "tested": True,
                "tested_at": datetime.now().isoformat(),
                "test_success": exec_result.get('success', False),
                "test_in_progress": False,
                
                # === Execution Metrics ===
                "execution_time": exec_result.get('execution_time', 0),
                "return_code": exec_result.get('return_code', -1),
                "timeout": exec_result.get('timeout', False),
                "executed_on": "trainium",
                "instance_type": os.getenv('TRAINIUM_INSTANCE_TYPE', 'trn1.2xlarge'),
                
                # === Performance Metrics (if provided by Trainium) ===
                "peak_memory_mb": exec_result.get('peak_memory_mb'),
                "neuron_core_utilization": exec_result.get('neuron_core_utilization'),
                "throughput_samples_per_sec": exec_result.get('throughput_samples_per_sec'),
                "training_loss": exec_result.get('training_loss'),
                "validation_accuracy": exec_result.get('validation_accuracy'),
                
                # === S3 Artifact References ===
                "outputs_s3_bucket": OUTPUTS_BUCKET,
                "stdout_s3_key": stdout_key,
                "stderr_s3_key": stderr_key,
                "artifacts_s3_prefix": f"{paper_id}/outputs/",
                "plots_s3_keys": plots_keys if plots_keys else None,
                "logs_s3_key": f"{paper_id}/outputs/execution.log",
                
                # === Dataset Information ===
                "dataset_name": exec_result.get('dataset_name'),
                "dataset_size_mb": exec_result.get('dataset_size_mb'),
                "dataset_s3_key": exec_result.get('dataset_s3_key'),
                "dataset_download_time": exec_result.get('dataset_download_time'),
                
                # === Code Quality Metrics ===
                "lines_of_code": exec_result.get('lines_of_code'),
                "has_training_loop": exec_result.get('has_training_loop'),
                "has_evaluation": exec_result.get('has_evaluation'),
                "uses_distributed_training": exec_result.get('uses_distributed_training'),
                "pytorch_version": exec_result.get('pytorch_version', '2.1.0'),
                
                # === Cost Tracking ===
                "trainium_hours": execution_hours,
                "estimated_compute_cost": round(estimated_cost, 2),
                
                # === Test Attempts (increment if exists) ===
                "test_attempts": 1,  # Will be incremented if retrying
                "last_test_attempt": datetime.now().isoformat()
            }
            
            # Add error info if execution failed
            if not exec_result.get('success', False):
                test_results['has_errors'] = True
                test_results['error_message'] = exec_result.get('error_message', 'Unknown error')
                test_results['error_type'] = exec_result.get('error_type', 'unknown')
            else:
                test_results['has_errors'] = False
            
            # Update OpenSearch with comprehensive results
            update_opensearch(paper_id, test_results)
            
            success_status = "✅ SUCCESS" if exec_result.get('success', False) else "❌ FAILED"
            logger.info(
                f"{success_status} | {paper_id} | "
                f"Time: {exec_result.get('execution_time', 0):.1f}s | "
                f"Cost: ${estimated_cost:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error processing results for paper {paper_id}: {e}")
            # Update OpenSearch with processing error
            try:
                update_opensearch(paper_id, {
                    "tested": True,
                    "tested_at": datetime.now().isoformat(),
                    "test_success": False,
                    "has_errors": True,
                    "error_message": f"Error processing results: {str(e)}",
                    "error_type": "processing_error",
                    "executed_on": "trainium"
                })
            except Exception as update_error:
                logger.error(f"Failed to update OpenSearch with error for {paper_id}: {update_error}")

def lambda_handler(event, context):
    """
    Lambda handler triggered by SQS messages.
    Processes batches of up to 10 messages at a time.
    """
    try:
        logger.info(f"PapersCodeTester invoked with {len(event['Records'])} messages")
        
        # Build batch data from SQS messages
        batch_data = []
        
        for record in event['Records']:
            message_body = json.loads(record['body'])
            
            paper_id = message_body['paper_id']
            paper_title = message_body['paper_title']
            s3_bucket = message_body['s3_bucket']
            s3_code_key = message_body['s3_code_key']
            
            logger.info(f"Processing paper: {paper_title} ({paper_id})")
            
            # Download code from S3
            try:
                code = download_code_from_s3(s3_bucket, s3_code_key)
                
                batch_data.append({
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "code": code,
                    "s3_bucket": s3_bucket,
                    "s3_code_key": s3_code_key
                })
            except Exception as e:
                logger.error(f"Error downloading code for {paper_id}: {e}")
                # Update OpenSearch with download error
                try:
                    update_opensearch(paper_id, {
                        "tested": True,
                        "tested_at": datetime.now().isoformat(),
                        "success": False,
                        "error_message": f"Failed to download code from S3: {str(e)}",
                        "error_type": "download_error"
                    })
                except Exception as update_error:
                    logger.error(f"Failed to update OpenSearch for {paper_id}: {update_error}")
        
        if not batch_data:
            logger.warning("No valid code files to process")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No valid code files to process"
                })
            }
        
        logger.info(f"Sending batch of {len(batch_data)} papers to Trainium instance")
        
        # Send batch to Trainium for execution
        trainium_results = send_batch_to_trainium(batch_data)
        
        # Process and save results
        process_batch_results(batch_data, trainium_results)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Processed {len(batch_data)} papers",
                "papers_processed": len(batch_data),
                "executed_on": "trainium"
            })
        }
        
    except Exception as e:
        logger.error(f"Error in PapersCodeTester: {e}")
        raise
