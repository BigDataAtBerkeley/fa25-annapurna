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
from typing import Dict, Any, List, Optional
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
ec2_client = boto3.client('ec2', region_name='us-east-2')  # Trainium is in us-east-2
session = boto3.Session()

OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'research-papers-v2')
OUTPUTS_BUCKET = os.getenv('OUTPUTS_BUCKET', 'papers-test-outputs')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

TRAINIUM_ENDPOINT = os.getenv('TRAINIUM_ENDPOINT')  
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
TRAINIUM_TIMEOUT = int(os.getenv('TRAINIUM_TIMEOUT', '1800'))  # 30 minutes (increased for Neuron compilation)

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

def find_instance_by_elastic_ip(elastic_ip: str) -> Optional[str]:
    """
    Find EC2 instance ID by Elastic IP address using describe_addresses.
    This works even when the instance is stopped (Elastic IP remains associated).
    Returns instance ID or None if not found.
    """
    try:
        # Extract IP from endpoint (remove http:// and port)
        ip = elastic_ip.replace('http://', '').replace('https://', '').split(':')[0]
        
        # Use describe_addresses - this works even when instance is stopped
        # Elastic IPs remain associated with instances even when stopped
        response = ec2_client.describe_addresses(PublicIps=[ip])
        
        if response['Addresses']:
            address = response['Addresses'][0]
            instance_id = address.get('InstanceId')
            
            if instance_id:
                logger.info(f"Found Trainium instance {instance_id} with Elastic IP {ip}")
                return instance_id
            else:
                # If no instance ID, check network interface
                network_interface_id = address.get('NetworkInterfaceId')
                if network_interface_id:
                    try:
                        ni_response = ec2_client.describe_network_interfaces(
                            NetworkInterfaceIds=[network_interface_id]
                        )
                        if ni_response['NetworkInterfaces']:
                            instance_id = ni_response['NetworkInterfaces'][0].get('Attachment', {}).get('InstanceId')
                            if instance_id:
                                logger.info(f"Found Trainium instance {instance_id} with Elastic IP {ip} (via network interface)")
                                return instance_id
                    except Exception as ni_err:
                        logger.debug(f"Could not get instance from network interface: {ni_err}")
        
        logger.warning(f"No Trainium instance found with Elastic IP {ip}")
        return None
    except Exception as e:
        logger.error(f"Error finding instance by Elastic IP: {e}")
        return None

def ensure_trainium_running() -> bool:
    """
    Ensure Trainium instance is running.
    Uses Elastic IP from TRAINIUM_ENDPOINT to find instance (works even when stopped).
    Starts instance if stopped.
    Returns True if instance is running or accessible, False otherwise.
    """
    if not TRAINIUM_ENDPOINT:
        logger.error("TRAINIUM_ENDPOINT not set - cannot connect to Trainium")
        return False
    
    logger.info(f"Using Trainium endpoint: {TRAINIUM_ENDPOINT} (Elastic IP - doesn't change)")
    
    # Find instance by Elastic IP (works even when instance is stopped)
    instance_id = find_instance_by_elastic_ip(TRAINIUM_ENDPOINT)
    
    if instance_id:
        try:
            response = ec2_client.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            state = instance['State']['Name']
            
            if state == 'running':
                logger.info(f"Trainium instance {instance_id} is running")
                # Do health check
                try:
                    health_response = requests.get(f"{TRAINIUM_ENDPOINT}/health", timeout=5)
                    if health_response.status_code == 200:
                        logger.info("Trainium health check passed")
                    else:
                        logger.warning(f"Trainium health check returned status {health_response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Could not reach Trainium health endpoint: {e}")
                return True
            elif state == 'stopped':
                logger.info(f"Starting Trainium instance {instance_id}...")
                ec2_client.start_instances(InstanceIds=[instance_id])
                
                # Wait for instance to be running
                waiter = ec2_client.get_waiter('instance_running')
                waiter.wait(InstanceIds=[instance_id])
                
                # Additional wait for services to start (Flask app, etc.)
                logger.info("Waiting 60 seconds for Trainium services to start...")
                time.sleep(60)
                logger.info(f"Trainium instance {instance_id} is now running")
                return True
            else:
                logger.warning(f"Trainium instance {instance_id} is in state: {state}")
                return False
        except Exception as e:
            logger.error(f"Error managing Trainium instance: {e}")
            return False
    else:
        # Instance not found by Elastic IP - assume it's running and accessible
        logger.warning("Could not find instance by Elastic IP. Assuming instance is running and accessible.")
        try:
            health_response = requests.get(f"{TRAINIUM_ENDPOINT}/health", timeout=5)
            if health_response.status_code == 200:
                logger.info("Trainium health check passed")
                return True
        except requests.exceptions.RequestException:
            pass
        return True

def download_code_from_s3(bucket: str, key: str) -> str:
    """Download code from S3"""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

def save_output_to_s3(paper_id: str, filename: str, content) -> str:
    """Save output to S3 and return the key
    
    Args:
        paper_id: Paper ID
        filename: Filename (can include subdirectories)
        content: Content to save (str, bytes, or base64 string)
    """
    key = f"{paper_id}/outputs/{filename}"
    
    # Handle base64 encoded content (for plots)
    if isinstance(content, str) and content.startswith('data:image'):
        import base64
        header, encoded = content.split(',', 1)
        body = base64.b64decode(encoded)
    elif isinstance(content, bytes):
        body = content
    else:
        body = content.encode('utf-8')
    
    s3_client.put_object(
        Bucket=OUTPUTS_BUCKET,
        Key=key,
        Body=body
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
            timeout=TRAINIUM_TIMEOUT + 30  
        )
        
        response.raise_for_status()
        results = response.json()
        
        logger.info(f"Received results from Trainium for {len(results.get('results', []))} papers")
        
        # Log detailed results for debugging
        for paper_id, result in results.get('results', {}).items():
            if not result.get('success', False):
                logger.error(f"Trainium execution failed for {paper_id}:")
                logger.error(f"  Error message: {result.get('error_message', 'N/A')}")
                logger.error(f"  Error type: {result.get('error_type', 'N/A')}")
                logger.error(f"  Return code: {result.get('return_code', 'N/A')}")
                if result.get('stderr'):
                    logger.error(f"  Stderr (first 500 chars): {result.get('stderr', '')[:500]}")
                if result.get('stdout'):
                    stdout_lines = result.get('stdout', '').split('\n')
                    logger.error(f"  Stdout (last 10 lines): {chr(10).join(stdout_lines[-10:])}")
        
        return results
        
    except requests.exceptions.Timeout:
        logger.error(f"Trainium execution timed out after {TRAINIUM_TIMEOUT} seconds")
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
        
        # If execution failed, save error outputs and update OpenSearch
        if not exec_result.get('success', False):
            error_msg = exec_result.get('error_message', exec_result.get('error', 'Unknown error'))
            logger.error(f"Execution failed for paper {paper_id}: {error_msg}")
            
            # Save stdout/stderr even on failure so we can debug
            try:
                stdout_key = save_output_to_s3(paper_id, 'stdout.log', exec_result.get('stdout', ''))
                stderr_key = save_output_to_s3(paper_id, 'stderr.log', exec_result.get('stderr', ''))
            except Exception as save_error:
                logger.warning(f"Failed to save error outputs for {paper_id}: {save_error}")
                stdout_key = None
                stderr_key = None
            
            # Update OpenSearch with failure details
            try:
                update_opensearch(paper_id, {
                    "tested": True,
                    "tested_at": datetime.now().isoformat(),
                    "test_success": False,
                    "has_errors": True,
                    "error_message": error_msg,
                    "error_type": exec_result.get('error_type', 'execution_error'),
                    "executed_on": "trainium",
                    "execution_time": exec_result.get('execution_time'),
                    "return_code": exec_result.get('return_code', -1),
                    "timeout": exec_result.get('timeout', False),
                    "outputs_s3_bucket": OUTPUTS_BUCKET,
                    "stdout_s3_key": stdout_key,
                    "stderr_s3_key": stderr_key,
                    "artifacts_s3_prefix": f"{paper_id}/outputs/"
                })
            except Exception as update_error:
                logger.error(f"Failed to update OpenSearch for {paper_id}: {update_error}")
            continue  # Dump this paper, continue with next
        
        try:
            stdout_key = save_output_to_s3(paper_id, 'stdout.log', exec_result.get('stdout', ''))
            stderr_key = save_output_to_s3(paper_id, 'stderr.log', exec_result.get('stderr', ''))
    
            plots_keys = []
            if 'plots' in exec_result:
                for plot_name, plot_data in exec_result.get('plots', {}).items():
                    try:
                        # Handle base64 encoded plots
                        if isinstance(plot_data, str) and plot_data.startswith('data:image'):
                            # Extract base64 data from data URI
                            import base64
                            header, encoded = plot_data.split(',', 1)
                            plot_data = base64.b64decode(encoded)
                        
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
            
            success_status = "SUCCESS" if exec_result.get('success', False) else "FAILED"
            logger.info(
                f"{success_status} | {paper_id} | "
                f"Time: {exec_result.get('execution_time', 0):.1f}s | "
                f"Cost: ${estimated_cost:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error processing results for paper {paper_id}: {e}")
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
        batch_item_failures = []  # For SQS partial batch failure handling
        
        for record in event['Records']:
            try:
                message_body = json.loads(record['body'])
                
                paper_id = message_body['paper_id']
                paper_title = message_body.get('paper_title', 'Unknown')
                s3_bucket = message_body['s3_bucket']
                s3_code_key = message_body['s3_code_key']
                message_id = record.get('messageId')  # Store messageId for error handling
                
                logger.info(f"Processing paper: {paper_title} ({paper_id})")
                
                # Download code from S3
                try:
                    code = download_code_from_s3(s3_bucket, s3_code_key)
                    
                    batch_data.append({
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "code": code,
                        "s3_bucket": s3_bucket,
                        "s3_code_key": s3_code_key,
                        "message_id": message_id  # Store for proper error handling
                    })
                except Exception as e:
                    logger.error(f"Error downloading code for {paper_id}: {e}")
                    # Mark this record as failed for SQS partial batch failure handling
                    batch_item_failures.append({
                        "itemIdentifier": record.get('messageId', record.get('receiptHandle', ''))
                    })
                    # Update OpenSearch with error
                    try:
                        update_opensearch(paper_id, {
                            "tested": True,
                            "tested_at": datetime.now().isoformat(),
                            "test_success": False,
                            "error_message": f"Failed to download code from S3: {str(e)}",
                            "error_type": "download_error"
                        })
                    except Exception as update_error:
                        logger.error(f"Failed to update OpenSearch for {paper_id}: {update_error}")
                        
            except Exception as e:
                logger.error(f"Error processing SQS record: {e}")
                # Mark this record as failed
                batch_item_failures.append({
                    "itemIdentifier": record.get('messageId', record.get('receiptHandle', ''))
                })
        
        if not batch_data:
            logger.warning("No valid code files to process after downloading from S3")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No valid code files to process",
                    "batchItemFailures": batch_item_failures
                })
            }
        
        logger.info(f"Sending batch of {len(batch_data)} papers to Trainium instance")
        
        # Send batch to Trainium for execution
        try:
            trainium_results = send_batch_to_trainium(batch_data)
            
            # Process and save results
            process_batch_results(batch_data, trainium_results)
        except Exception as e:
            logger.error(f"Error executing batch on Trainium: {e}")
            # Mark all batch items as failed
            for item in batch_data:
                batch_item_failures.append({
                    "itemIdentifier": item.get('message_id', item.get('paper_id'))  # Use messageId for proper SQS retry
                })
                # Update OpenSearch with error
                try:
                    update_opensearch(item['paper_id'], {
                        "tested": True,
                        "tested_at": datetime.now().isoformat(),
                        "test_success": False,
                        "error_message": f"Batch execution error: {str(e)}",
                        "error_type": "batch_execution_error"
                    })
                except Exception as update_error:
                    logger.error(f"Failed to update OpenSearch for {item['paper_id']}: {update_error}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"Processed {len(batch_data)} papers",
                "papers_processed": len(batch_data),
                "executed_on": "trainium",
                "batchItemFailures": batch_item_failures if batch_item_failures else []
            })
        }
        
    except Exception as e:
        logger.error(f"Error in PapersCodeTester: {e}")
        raise
