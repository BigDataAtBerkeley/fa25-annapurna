#!/usr/bin/env python3
"""
Main Pipeline for Delivery

This pipeline processes one paper at a time:
1. Retrieves paper from OpenSearch
2. Generates PyTorch code using Bedrock (with Neuron SDK requirements)
3. Executes on Trainium
4. Monitors and saves results at each step

Note: This script runs code generation locally using ChunkedPyTorchGenerator.
The same code generation logic is also available in AWS Lambda:
- PapersCodeGenerator-container (container-based, recommended - fixes pymupdf issues)
- PapersCodeGenerator (old zip-based, has pymupdf dependency issues)

The Lambda function is triggered by SQS queue (code-evaluation.fifo) and uses
the same ChunkedPyTorchGenerator code for consistency.
"""

import os
import sys
import json
import logging
import time
import subprocess
import boto3
import requests
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Setup logging first (before imports that might fail)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_for_delivery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add code_gen to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code_gen'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trn_execute'))

from storage_utils import save_json, save_code

# Import SlackNotifier for initial paper notifications
try:
    from slack_notifier import SlackNotifier
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logger.warning("slack_notifier module not available - initial paper notifications will be disabled")

try:
    # Import chunked generator from code_gen subdirectory
    import importlib.util
    chunked_dir = os.path.join(os.path.dirname(__file__), 'code_gen')
    chunked_generator_path = os.path.join(chunked_dir, 'chunked_generator.py')
    
    if os.path.exists(chunked_generator_path):
        # Add the code_gen directory to path so it can import its dependencies
        sys.path.insert(0, chunked_dir)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code_gen'))
        
        # Load the chunked_generator module
        spec = importlib.util.spec_from_file_location("chunked_generator", chunked_generator_path)
        chunked_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chunked_module)
        ChunkedPyTorchGenerator = chunked_module.ChunkedPyTorchGenerator
        CHUNKED_AVAILABLE = True
        logger.info("✅ Chunked code generator loaded successfully")
    else:
        CHUNKED_AVAILABLE = False
        logger.warning(f"Chunked code generator not available (chunked_generator.py not found at {chunked_generator_path})")
except Exception as e:
    CHUNKED_AVAILABLE = False
    logger.warning(f"Chunked code generator not available: {e}")
    import traceback
    logger.debug(traceback.format_exc())

# EC2 and Trainium configuration
TRAINIUM_ENDPOINT = os.getenv('TRAINIUM_ENDPOINT')  # e.g., "http://1.2.3.4:8000"
TRAINIUM_INSTANCE_ID = os.getenv('TRAINIUM_INSTANCE_ID')
TRAINIUM_REGION = os.getenv('TRAINIUM_REGION', 'us-east-2')
TRAINIUM_TIMEOUT = int(os.getenv('TRAINIUM_TIMEOUT', '1800'))  # 30 minutes

# AWS clients for EC2 management
ec2_client = None
if TRAINIUM_INSTANCE_ID:
    ec2_client = boto3.client('ec2', region_name=TRAINIUM_REGION)

# Results directory structure - per-paper folders
RESULTS_DIR = Path('results')
# Directories will be created per-paper: results/{paper_id}/{step}/


def save_step_result(step_name: str, paper_id: str, data: Dict[str, Any], 
                     subdir: Optional[Path] = None) -> str:
    """
    Save result from a pipeline step to storage (local or S3).
    Uses per-paper folder structure: results/{paper_id}/{step_name}/
    
    Args:
        step_name: Name of the step (e.g., 'code-generation', 'code-review', 'trn-execution')
        paper_id: Paper ID
        data: Data to save
        subdir: Optional subdirectory within the step directory (not used in new structure)
        
    Returns:
        Path to saved file (local path or S3 key)
    """
    return save_json(paper_id, step_name, data)


def execute_code_via_http(paper_id: str, code: str, timeout: int = 1800, paper_title: Optional[str] = None, slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
    """
    Send code to Trainium for async execution via HTTP request.
    Execution runs asynchronously - this function only waits for acknowledgment.
    
    Args:
        paper_id: Paper ID
        code: Code to execute
        timeout: Execution timeout in seconds (for the actual execution, not HTTP request)
        paper_title: Paper title (optional)
        slack_thread_ts: Slack thread timestamp to reply in thread (optional)
        
    Returns:
        Dictionary with async acknowledgment (status: "running", job_id, status_url)
    """
    if not TRAINIUM_ENDPOINT:
        raise ValueError("TRAINIUM_ENDPOINT not set - cannot execute code")
    
    endpoint = f"{TRAINIUM_ENDPOINT}/execute"
    
    payload = {
        "paper_id": paper_id,
        "code": code,
        "timeout": timeout
    }
    
    if paper_title:
        payload["paper_title"] = paper_title
    
    if slack_thread_ts:
        payload["slack_thread_ts"] = slack_thread_ts
    
    logger.info(f"Sending execution request to {endpoint} (async execution)")
    
    # Retry logic for connection errors
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Use short timeout - we only need to get the acknowledgment that execution started
            # Execution itself runs asynchronously on Trainium
            http_timeout = 30  # 30 seconds should be enough to get acknowledgment
            if attempt == 0:
                logger.info(f"HTTP request timeout set to {http_timeout}s (waiting for execution acknowledgment)")
            
            response = requests.post(
                endpoint,
                json=payload,
                timeout=http_timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Verify we got an async acknowledgment
            if result.get("status") == "running" or result.get("job_id"):
                logger.info(f"✅ Execution acknowledged - job started (job_id: {result.get('job_id', 'N/A')})")
                return result
            else:
                # Unexpected response format - log warning but return it
                logger.warning(f"Unexpected response format - expected async acknowledgment but got: {result}")
                return result
                
        except requests.exceptions.Timeout as e:
            logger.error(f"HTTP request timed out after {http_timeout}s - failed to get execution acknowledgment")
            return {
                "success": False,
                "error_message": f"HTTP request timed out after {http_timeout}s. Failed to get execution acknowledgment.",
                "error_type": "http_timeout",
                "status": "unknown",
                "note": "Could not confirm if execution started. Check Trainium logs to verify."
            }
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            # Connection errors - retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"HTTP connection error (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"HTTP connection failed after {max_retries} attempts: {e}")
                return {
                    "success": False,
                    "error_message": f"HTTP connection failed after {max_retries} attempts: {str(e)}",
                    "error_type": "http_error",
                    "status": "unknown",
                    "note": "Could not connect to executor. Check executor status and network connectivity."
                }
        except requests.exceptions.RequestException as e:
            # Other request errors - don't retry
            logger.error(f"HTTP request to Trainium executor failed: {e}")
            return {
                "success": False,
                "error_message": f"HTTP request failed: {str(e)}",
                "error_type": "http_error",
                "status": "unknown"
            }


def find_instance_by_elastic_ip(elastic_ip: str) -> Optional[str]:
    """
    Find EC2 instance by Elastic IP address.
    
    Args:
        elastic_ip: Elastic IP address
        
    Returns:
        Instance ID if found, None otherwise
    """
    if not ec2_client:
        return None
    
    try:
        # Get all instances in the region
        response = ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'stopped', 'pending']}
            ]
        )
        
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                # Check Elastic IP associations
                network_interfaces = instance.get('NetworkInterfaces', [])
                for ni in network_interfaces:
                    association = ni.get('Association', {})
                    if association and association.get('PublicIp') == elastic_ip:
                        return instance['InstanceId']
        
        return None
    except Exception as e:
        logger.error(f"Error finding instance by Elastic IP: {e}")
        return None


def ensure_trainium_running() -> bool:
    """
    Ensure Trainium EC2 instance is running.
    Uses Elastic IP from TRAINIUM_ENDPOINT to find instance if TRAINIUM_INSTANCE_ID not set.
    Starts instance if stopped, waits for it to be ready.
    
    Returns:
        True if instance is running and accessible, False otherwise
    """
    # Try to find instance ID from Elastic IP if not set
    instance_id = TRAINIUM_INSTANCE_ID
    if not instance_id and TRAINIUM_ENDPOINT and ec2_client:
        # Extract IP from endpoint
        endpoint_ip = TRAINIUM_ENDPOINT.replace('http://', '').replace('https://', '').split(':')[0]
        logger.info(f"TRAINIUM_INSTANCE_ID not set, trying to find instance by Elastic IP: {endpoint_ip}")
        instance_id = find_instance_by_elastic_ip(endpoint_ip)
        if instance_id:
            logger.info(f"Found instance ID: {instance_id} for Elastic IP {endpoint_ip}")
        else:
            logger.warning(f"Could not find instance by Elastic IP {endpoint_ip}")
    
    if not instance_id or not ec2_client:
        logger.warning("TRAINIUM_INSTANCE_ID not set and could not find by Elastic IP - assuming Trainium executor is already running")
        # Still try to verify endpoint is accessible
        if TRAINIUM_ENDPOINT:
            try:
                response = requests.get(f"{TRAINIUM_ENDPOINT}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Trainium executor is accessible at {TRAINIUM_ENDPOINT}")
                    return True
            except Exception as e:
                logger.warning(f"Could not reach Trainium endpoint: {e}")
        return True
    
    try:
        response = ec2_client.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        state = instance['State']['Name']
        
        if state == 'running':
            logger.info(f"✅ Trainium instance {instance_id} is already running")
            # Verify endpoint is accessible
            if TRAINIUM_ENDPOINT:
                try:
                    response = requests.get(f"{TRAINIUM_ENDPOINT}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ Trainium executor is accessible at {TRAINIUM_ENDPOINT}")
                        return True
                    else:
                        logger.warning(f"⚠️ Trainium endpoint returned status {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not reach Trainium endpoint: {e}")
            return True
        elif state == 'stopped':
            logger.info(f"⏳ Starting Trainium instance {instance_id}...")
            ec2_client.start_instances(InstanceIds=[instance_id])
            
            # Wait for instance to be running
            logger.info("⏳ Waiting for instance to start...")
            waiter = ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[instance_id])
            
            # Get new IP address
            response = ec2_client.describe_instances(InstanceIds=[instance_id])
            instance = response['Reservations'][0]['Instances'][0]
            public_ip = instance.get('PublicIpAddress')
            
            if public_ip:
                logger.info(f"📌 Instance started with IP: {public_ip}")
                # Check if it's an Elastic IP (should match TRAINIUM_ENDPOINT)
                network_interfaces = instance.get('NetworkInterfaces', [])
                has_elastic_ip = any(ni.get('Association', {}).get('PublicIp') for ni in network_interfaces)
                if has_elastic_ip:
                    logger.info(f"✅ Instance has Elastic IP (static, won't change)")
                else:
                    logger.warning(f"⚠️ Instance IP may change on restart. Consider using Elastic IP.")
                    if not TRAINIUM_ENDPOINT:
                        logger.warning(f"⚠️ Update TRAINIUM_ENDPOINT to: http://{public_ip}:8000")
            
            # Wait for services to start (Flask app, etc.)
            logger.info("⏳ Waiting 60 seconds for Trainium services to start...")
            time.sleep(60)
            
            # Verify endpoint is accessible
            if TRAINIUM_ENDPOINT:
                max_retries = 10
                for i in range(max_retries):
                    try:
                        response = requests.get(f"{TRAINIUM_ENDPOINT}/health", timeout=5)
                        if response.status_code == 200:
                            logger.info(f"✅ Trainium executor is ready at {TRAINIUM_ENDPOINT}")
                            return True
                    except Exception as e:
                        if i < max_retries - 1:
                            logger.info(f"⏳ Waiting for executor to be ready... ({i+1}/{max_retries})")
                            time.sleep(10)
                        else:
                            logger.warning(f"⚠️ Could not reach Trainium endpoint after {max_retries} attempts: {e}")
            
            logger.info(f"✓ Trainium instance {instance_id} is now running")
            return True
        else:
            logger.warning(f"⚠️ Trainium instance {instance_id} is in state: {state}")
            return False
            
    except Exception as e:
        logger.error(f"Error managing Trainium instance: {e}")
        return False


def save_code_file(paper_id: str, code: str, step: str) -> str:
    """Save code to storage (local or S3).
    Uses per-paper folder structure: results/{paper_id}/{step}/"""
    return save_code(paper_id, step, code)


def process_paper(paper_id: str, generator) -> Dict[str, Any]:
    """
    Process a single paper through the entire pipeline.
    
    Args:
        paper_id: OpenSearch document ID
        generator: ChunkedPyTorchGenerator instance (PDF-only with smart chunking)
        
    Returns:
        Dictionary with complete pipeline results
    """
    pipeline_start = time.time()
    pipeline_results = {
        "paper_id": paper_id,
        "pipeline_start": datetime.now().isoformat(),
        "steps": {}
    }
    
    try:
        # Step 1: Retrieve paper from OpenSearch
        logger.info(f"Step 1: Retrieving paper {paper_id} from OpenSearch...")
        step1_start = time.time()
        
        paper = generator.opensearch_client.get_paper_by_id(paper_id)
        if not paper:
            pipeline_results["success"] = False
            pipeline_results["error"] = f"Paper {paper_id} not found in OpenSearch"
            return pipeline_results
        
        paper_summary = generator.opensearch_client.get_paper_summary(paper)
        paper_content = generator.opensearch_client.get_paper_content(paper)
        
        step1_result = {
            "paper_id": paper_id,
            "paper_title": paper.get('title', 'Unknown'),
            "paper_authors": paper.get('authors', []),
            "has_content": paper_content is not None,
            "content_length": len(paper_content) if paper_content else 0,
            "retrieval_time": time.time() - step1_start
        }
        
        save_step_result('paper-retrieval', paper_id, {
            "paper": paper,
            "summary": paper_summary,
            "metadata": step1_result
        })
        
        pipeline_results["steps"]["paper_retrieval"] = step1_result
        logger.info(f"✅ Paper retrieved: {paper.get('title', 'Unknown')}")
        
        # Send initial paper notification to Slack (creates thread)
        slack_thread_ts = None
        if SLACK_AVAILABLE:
            try:
                filtered_paper = {k: v for k, v in paper.items() if k != 'embeddings'}
                filtered_paper['_id'] = paper_id
                slack_notifier = SlackNotifier()
                slack_thread_ts = slack_notifier.send_paper_info(filtered_paper)
                if slack_thread_ts:
                    logger.info(f"✅ Sent initial paper notification to Slack (thread_ts: {slack_thread_ts})")
                else:
                    logger.warning("⚠️ Failed to send initial paper notification to Slack")
            except Exception as e:
                logger.warning(f"⚠️ Error sending initial paper notification to Slack: {e}")
        
        # Step 2: Generate code
        logger.info(f"Step 2: Generating PyTorch code for paper {paper_id}...")
        step2_start = time.time()
        
        # Handle different generator interfaces
        if hasattr(generator, 'generate_code_for_paper'):
            # Check if it's chunked generator (no include_full_content param) or standard
            import inspect
            sig = inspect.signature(generator.generate_code_for_paper)
            if 'include_full_content' in sig.parameters:
                # Standard generator
                code_gen_result = generator.generate_code_for_paper(paper_id, include_full_content=True)
            else:
                # Chunked generator
                code_gen_result = generator.generate_code_for_paper(paper_id)
        else:
            pipeline_results["success"] = False
            pipeline_results["error"] = "Generator does not have generate_code_for_paper method"
            return pipeline_results
        
        if not code_gen_result.get("success") or not code_gen_result.get("code"):
            pipeline_results["success"] = False
            pipeline_results["error"] = code_gen_result.get("error", "Code generation failed")
            pipeline_results["steps"]["code_generation"] = {
                "success": False,
                "error": code_gen_result.get("error"),
                "generation_time": time.time() - step2_start
            }
            return pipeline_results
        
        # Save generated code
        code_file = save_code_file(paper_id, code_gen_result["code"], 'code-generation')
        
        step2_result = {
            "success": True,
            "paper_id": paper_id,
            "paper_title": code_gen_result.get("paper_title"),
            "code_length": len(code_gen_result["code"]),
            "model_used": code_gen_result.get("model_used"),
            "recommended_dataset": code_gen_result.get("recommended_dataset"),
            "code_file": code_file,
            "generation_time": time.time() - step2_start,
            "truncated": code_gen_result.get("truncated", False)
        }
        
        save_step_result('code-generation', paper_id, {
            "code": code_gen_result["code"],
            "metadata": step2_result,
            "dataset_recommendations": code_gen_result.get("dataset_recommendations"),
            "explanation": code_gen_result.get("explanation")
        })
        
        pipeline_results["steps"]["code_generation"] = step2_result
        logger.info(f"✅ Code generated ({len(code_gen_result['code'])} chars)")
        
        # Step 2.5: Code Reviewer 0 - Proactive TRN compatibility fixes
        logger.info(f"Step 2.5: Code Reviewer 0 - Proactively fixing TRN compatibility issues...")
        step2_5_start = time.time()
        
        initial_code = code_gen_result["code"]
        reviewed_code = initial_code
        
        if TRAINIUM_ENDPOINT:
            try:
                review_endpoint = f"{TRAINIUM_ENDPOINT}/code_review_0"
                review_payload = {
                    "paper_id": paper_id,
                    "code": initial_code,
                    "paper_summary": paper_summary
                }
                
                logger.info(f"Sending code to Code Reviewer 0 at {review_endpoint}")
                review_response = requests.post(
                    review_endpoint,
                    json=review_payload,
                    timeout=300  # 5 minute timeout for review
                )
                review_response.raise_for_status()
                review_result = review_response.json()
                
                if review_result.get("success") and review_result.get("code"):
                    reviewed_code = review_result["code"]
                    fixes_summary = review_result.get("fixes_summary", [])
                    code_changed = review_result.get("code_changed", False)
                    
                    if code_changed:
                        logger.info(f"✅ Code Reviewer 0: Fixed TRN compatibility issues")
                        logger.info(f"   Fixes: {', '.join(fixes_summary) if isinstance(fixes_summary, list) else fixes_summary}")
                    else:
                        logger.info(f"ℹ️ Code Reviewer 0: No changes needed - code is already TRN compatible")
                    
                    step2_5_result = {
                        "success": True,
                        "code_changed": code_changed,
                        "fixes_summary": fixes_summary,
                        "review_time": time.time() - step2_5_start
                    }
                else:
                    logger.warning(f"⚠️ Code Reviewer 0 failed: {review_result.get('error', 'Unknown error')}")
                    logger.warning(f"   Using original code without TRN compatibility fixes")
                    step2_5_result = {
                        "success": False,
                        "error": review_result.get("error", "Code review failed"),
                        "code_changed": False,
                        "review_time": time.time() - step2_5_start
                    }
            except Exception as e:
                logger.warning(f"⚠️ Code Reviewer 0 error: {e}")
                logger.warning(f"   Using original code without TRN compatibility fixes")
                step2_5_result = {
                    "success": False,
                    "error": str(e),
                    "code_changed": False,
                    "review_time": time.time() - step2_5_start
                }
        else:
            logger.warning("⚠️ TRAINIUM_ENDPOINT not set - skipping Code Reviewer 0")
            step2_5_result = {
                "success": False,
                "error": "TRAINIUM_ENDPOINT not set",
                "code_changed": False,
                "review_time": 0,
                "skipped": True
            }
        
        save_step_result('code-review-0', paper_id, {
            "original_code_length": len(initial_code),
            "reviewed_code_length": len(reviewed_code),
            "metadata": step2_5_result
        })
        
        pipeline_results["steps"]["code_review_0"] = step2_5_result
        
        # Update code to use reviewed version
        code_gen_result["code"] = reviewed_code
        
        # Step 3: Execute on Trainium via HTTP
        if not TRAINIUM_ENDPOINT:
            logger.warning("⚠️ TRAINIUM_ENDPOINT not set - skipping execution")
            logger.warning("   Code generation completed successfully")
            pipeline_results["steps"]["trn_execution"] = {
                "success": False,
                "error": "TRAINIUM_ENDPOINT not set",
                "execution_time_seconds": 0,
                "skipped": True
            }
            pipeline_results["success"] = True  # Code generation succeeded
            return pipeline_results
        
        # Step 3a: Ensure Trainium is running
        logger.info(f"Step 3a: Ensuring Trainium instance is running...")
        if not ensure_trainium_running():
            pipeline_results["success"] = False
            pipeline_results["error"] = "Failed to start or connect to Trainium instance"
            pipeline_results["steps"]["trn_execution"] = {
                "success": False,
                "error": "Trainium instance not available",
                "execution_time_seconds": 0
            }
            return pipeline_results
        logger.info(f"✅ Trainium instance is ready")
        
        # Step 3b: Execute on Trainium via HTTP (async execution)
        logger.info(f"Step 3b: Sending code to Trainium for async execution (paper {paper_id})...")
        step3_start = time.time()
        
        # Execution timeout for the actual execution (not HTTP request)
        execution_timeout = int(os.getenv('TRAINIUM_EXECUTION_TIMEOUT', '3600'))  # 60 minutes default
        execution_result = execute_code_via_http(
            paper_id=paper_id,
            code=code_gen_result["code"],
            timeout=execution_timeout,
            paper_title=code_gen_result.get("paper_title"),
            slack_thread_ts=slack_thread_ts
        )
        
        # Execution is always async - expect immediate acknowledgment
        if execution_result.get("status") == "running" or execution_result.get("job_id"):
            # Successfully received async acknowledgment
            logger.info(f"✅ Execution acknowledged - running asynchronously for {paper_id}")
            logger.info(f"   Job ID: {execution_result.get('job_id', paper_id)}")
            logger.info(f"   Status URL: {execution_result.get('status_url', 'N/A')}")
            
            step3_result = {
                "success": True,  # Job started successfully
                "status": "running",
                "job_id": execution_result.get("job_id", paper_id),
                "status_url": execution_result.get("status_url"),
                "execution_time": None,  # Not available yet - execution still running
                "return_code": None,  # Not available yet - execution still running
                "timeout": False,
                "metrics": {},
                "execution_time_seconds": time.time() - step3_start,
                "note": "Execution started asynchronously. Use status_url to check completion status."
            }
        else:
            # Failed to get acknowledgment or error response
            logger.error(f"❌ Failed to get execution acknowledgment for {paper_id}")
            step3_result = {
                "success": False,
                "status": execution_result.get("status", "unknown"),
                "error": execution_result.get("error_message", "Failed to start execution"),
                "error_type": execution_result.get("error_type", "unknown"),
                "execution_time": None,
                "return_code": None,
                "execution_time_seconds": time.time() - step3_start
            }
        
        # Save execution results
        save_step_result('trn-execution', paper_id, {
            "execution_result": execution_result,
            "metadata": step3_result,
            "stdout": execution_result.get("stdout", ""),
            "stderr": execution_result.get("stderr", "")
        })
        
        # Save metrics separately
        if step3_result.get("metrics"):
            save_step_result('metrics', paper_id, {
                "paper_id": paper_id,
                "paper_title": code_gen_result.get("paper_title"),
                "metrics": step3_result["metrics"],
                "execution_time": step3_result["execution_time"],
                "success": step3_result["success"]
            })
        
        # Save profiler results if available
        profiler_info = execution_result.get("profiler")
        if profiler_info and profiler_info.get("profiler_enabled"):
            profiler_output_dir = profiler_info.get("profiler_output_dir")
            profiler_files = profiler_info.get("profiler_files", [])
            perfetto_file_path = profiler_info.get("perfetto_file")  # Path on Trainium
            
            if profiler_output_dir and profiler_files:
                # Save profiler metadata
                save_step_result('profiler', paper_id, {
                    "paper_id": paper_id,
                    "paper_title": code_gen_result.get("paper_title"),
                    "profiler_output_dir": profiler_output_dir,
                    "profiler_files": profiler_files,
                    "profiler_enabled": True,
                    "perfetto_file": perfetto_file_path,
                    "note": f"Profiler output files are on Trainium instance at: {profiler_output_dir}"
                })
                logger.info(f"✅ Profiler results available: {len(profiler_files)} files in {profiler_output_dir}")
                
                # Download Perfetto file from Trainium if available
                if perfetto_file_path:
                    try:
                        # Create profiler_file directory in paper's results folder
                        paper_dir = RESULTS_DIR / paper_id
                        profiler_file_dir = paper_dir / 'profiler_file'
                        profiler_file_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Download from Trainium using SCP
                        trainium_host = TRAINIUM_ENDPOINT.replace('http://', '').replace('https://', '').split(':')[0]
                        ssh_key = os.getenv('SSH_KEY', '~/.ssh/trainium-deploy-key.pem')
                        
                        local_perfetto_file = profiler_file_dir / f"{paper_id}_profile.pftrace"
                        
                        scp_cmd = [
                            'scp',
                            '-i', os.path.expanduser(ssh_key),
                            '-o', 'StrictHostKeyChecking=no',
                            f'ec2-user@{trainium_host}:{perfetto_file_path}',
                            str(local_perfetto_file)
                        ]
                        
                        logger.info(f"Downloading Perfetto file from Trainium: {perfetto_file_path}")
                        download_result = subprocess.run(
                            scp_cmd,
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minute timeout
                        )
                        
                        if download_result.returncode == 0 and local_perfetto_file.exists():
                            file_size = local_perfetto_file.stat().st_size / (1024 * 1024)
                            logger.info(f"✓ Downloaded Perfetto file to {local_perfetto_file} ({file_size:.2f} MB)")
                        else:
                            logger.warning(f"Failed to download Perfetto file: {download_result.stderr}")
                    except Exception as e:
                        logger.warning(f"Failed to download Perfetto file from Trainium: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
        
        pipeline_results["steps"]["trn_execution"] = step3_result
        
        # Log execution status
        if step3_result.get("success"):
            logger.info(f"⏳ Execution started asynchronously for {paper_id}")
            logger.info(f"   Job ID: {step3_result.get('job_id')}")
            logger.info(f"   Status URL: {step3_result.get('status_url')}")
            logger.info(f"   Execution is running in background - check status endpoint for completion")
        else:
            error_msg = step3_result.get("error", "Unknown error")
            logger.error(f"❌ Failed to start execution: {error_msg}")
        
        # Overall pipeline result
        # Success means code generation succeeded and execution was acknowledged (started)
        pipeline_results["success"] = (
            step2_result.get("success") and 
            step3_result.get("success")  # True if execution was acknowledged/started
        )
        pipeline_results["pipeline_end"] = datetime.now().isoformat()
        pipeline_results["total_pipeline_time"] = time.time() - pipeline_start
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Error processing paper {paper_id}: {e}", exc_info=True)
        pipeline_results["success"] = False
        pipeline_results["error"] = str(e)
        pipeline_results["pipeline_end"] = datetime.now().isoformat()
        pipeline_results["total_pipeline_time"] = time.time() - pipeline_start
        return pipeline_results


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process papers through the delivery pipeline')
    parser.add_argument('--paper-id', help='Process a specific paper by ID')
    parser.add_argument('--max-papers', type=int, default=1, 
                       help='Maximum number of papers to process (default: 1)')
    parser.add_argument('--query', help='OpenSearch query to find papers (JSON format)')
    parser.add_argument('--recent-days', type=int, default=30,
                       help='Process recent papers from last N days')
    parser.add_argument('--random', action='store_true',
                       help='Get random papers instead of recent papers')
    parser.add_argument('--enable-execution-testing', action='store_true',
                       help='Enable execution testing during code review (tests code on Trainium each iteration)')
    
    args = parser.parse_args()
    
    # Set environment variable for chunked generator if needed (must be before initialization)
    if args.enable_execution_testing:
        os.environ['ENABLE_EXECUTION_TESTING'] = 'true'
        logger.info("✅ Execution testing enabled - code will be tested on Trainium during review")
    
    # Initialize generator - always use ChunkedPyTorchGenerator (PDF-only with smart chunking)
    if not CHUNKED_AVAILABLE:
        logger.error("❌ Chunked code generator not available")
        logger.error("   Make sure chunked-code-gen module is properly set up")
        return
    
    logger.info("Initializing Chunked PyTorch Code Generator (PDF-only with smart chunking)...")
    generator = ChunkedPyTorchGenerator(
        batch_size=8,  # Group 8 chunk summaries into each batch for hierarchical summarization
        use_smart_pdf_chunking=True,  # Enable smart chunking to prioritize relevant sections (filters appendix)
        max_pdf_chunks=15,  # Maximum number of PDF chunks to process (prioritizes abstract, formulas, diagrams)
        pages_per_pdf_chunk=2  # 2 pages per PDF chunk
    )
    logger.info("✅ Using PDF chunking with smart relevance filtering")
    
    # Ensure Trainium is ready before processing papers
    if TRAINIUM_ENDPOINT:
        logger.info("Checking Trainium instance status...")
        if not ensure_trainium_running():
            logger.error("❌ Cannot proceed - Trainium instance is not available")
            logger.error("   Please ensure TRAINIUM_INSTANCE_ID is set or Trainium executor is running")
            return
    else:
        logger.warning("⚠️ TRAINIUM_ENDPOINT not set - will only generate and review code")
        logger.warning("   Set TRAINIUM_ENDPOINT in .env to enable execution")
    
    # Get papers to process
    paper_ids = []
    
    if args.paper_id:
        paper_ids = [args.paper_id]
    elif args.query:
        query = json.loads(args.query)
        papers = generator.opensearch_client.search_papers(query, size=args.max_papers)
        paper_ids = [p.get('_id') for p in papers if p.get('_id')]
    elif args.random:
        # Get random papers
        logger.info(f"Selecting {args.max_papers} random papers...")
        papers = generator.opensearch_client.get_random_papers(size=args.max_papers)
        paper_ids = [p.get('_id') for p in papers if p.get('_id')]
        if paper_ids:
            logger.info(f"Selected random papers: {', '.join(paper_ids)}")
    else:
        # Get recent papers (default behavior)
        papers = generator.opensearch_client.get_recent_papers(
            days=args.recent_days, 
            size=args.max_papers
        )
        paper_ids = [p.get('_id') for p in papers if p.get('_id')]
    
    if not paper_ids:
        logger.error("No papers found to process")
        return
    
    logger.info(f"Processing {len(paper_ids)} paper(s)...")
    
    # Process each paper
    all_results = []
    for i, paper_id in enumerate(paper_ids, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing paper {i}/{len(paper_ids)}: {paper_id}")
        logger.info(f"{'='*80}\n")
        
        result = process_paper(paper_id, generator)
        all_results.append(result)
        
        # Save overall pipeline result
        # Save final result to paper's root directory
        save_step_result('', paper_id, result)
        
        # Brief summary
        if result.get("success"):
            logger.info(f"✅ Paper {paper_id} processed successfully")
        else:
            logger.error(f"❌ Paper {paper_id} failed: {result.get('error', 'Unknown error')}")
    
    # Final summary
    successful = sum(1 for r in all_results if r.get("success"))
    logger.info(f"\n{'='*80}")
    logger.info(f"Pipeline complete: {successful}/{len(all_results)} papers processed successfully")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()





