#!/usr/bin/env python3
"""
Main Pipeline for Delivery

This pipeline processes one paper at a time:
1. Retrieves paper from OpenSearch
2. Generates PyTorch code using Bedrock (with Neuron SDK requirements)
3. Reviews and fixes code
4. Executes on Trainium
5. Monitors and saves results at each step
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

# Add code-gen-for-deliv to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code-gen-for-deliv'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trn-execute-for-deliv'))

from pytorch_generator import PyTorchCodeGenerator
try:
    # Import chunked generator from chunked-code-gen subdirectory
    # Since the directory name has hyphens, we need to import the module directly
    import importlib.util
    chunked_dir = os.path.join(os.path.dirname(__file__), 'code-gen-for-deliv', 'chunked-code-gen')
    chunked_generator_path = os.path.join(chunked_dir, 'chunked_generator.py')
    
    if os.path.exists(chunked_generator_path):
        # Add the chunked-code-gen directory to path so it can import its dependencies
        sys.path.insert(0, chunked_dir)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code-gen-for-deliv'))
        
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
    Save result from a pipeline step to the results directory.
    Uses per-paper folder structure: results/{paper_id}/{step_name}/
    
    Args:
        step_name: Name of the step (e.g., 'code-generation', 'code-review', 'trn-execution')
        paper_id: Paper ID
        data: Data to save
        subdir: Optional subdirectory within the step directory (not used in new structure)
        
    Returns:
        Path to saved file
    """
    # Use per-paper folder structure: results/{paper_id}/{step_name}/
    paper_dir = RESULTS_DIR / paper_id
    save_dir = paper_dir / step_name if step_name else paper_dir
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{paper_id}_{timestamp}.json"
    filepath = save_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {step_name} result to {filepath}")
    return str(filepath)


def execute_code_via_http(paper_id: str, code: str, timeout: int = 1800, paper_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute code on Trainium via HTTP request to the executor endpoint.
    
    Args:
        paper_id: Paper ID
        code: Code to execute
        timeout: Execution timeout in seconds
        paper_title: Paper title (optional)
        
    Returns:
        Execution result dictionary
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
    
    logger.info(f"Sending execution request to {endpoint}")
    
    # Retry logic for connection errors
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Use longer timeout: execution_timeout + 5 minutes buffer for HTTP overhead
            # Neuron compilation can take 20-40+ minutes, so we need a long timeout
            http_timeout = timeout + 300  # Add 5 minute buffer
            if attempt == 0:
                logger.info(f"HTTP request timeout set to {http_timeout}s ({http_timeout/60:.1f} minutes)")
            
            response = requests.post(
                endpoint,
                json=payload,
                timeout=http_timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            logger.warning(f"HTTP request timed out after {http_timeout}s")
            logger.warning("Execution may still be running on Trainium - check Trainium logs to verify")
            return {
                "success": False,
                "error_message": f"HTTP request timed out after {http_timeout}s. Execution may still be running on Trainium.",
                "error_type": "http_timeout",
                "execution_time": 0,
                "return_code": -1,
                "note": "Check Trainium logs and CloudWatch metrics to verify if execution completed"
            }
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            # Connection errors - retry with exponential backoff
            # "Remote end closed connection without response" usually means:
            # 1. Flask app timed out waiting for execution (but execution may still be running)
            # 2. Flask app crashed
            # 3. Network issue
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"HTTP connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if "Remote end closed connection" in str(e):
                    logger.warning("⚠️ Flask server closed connection - execution may still be running on Trainium")
                    logger.warning("   Check Trainium logs/CloudWatch to verify if execution completed")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"HTTP connection failed after {max_retries} attempts: {e}")
                if "Remote end closed connection" in str(e):
                    logger.error("⚠️ Flask server closed connection - execution may still be running on Trainium")
                    logger.error("   Check Trainium instance logs and CloudWatch metrics to verify execution status")
                return {
                    "success": False,
                    "error_message": f"HTTP connection failed after {max_retries} attempts: {str(e)}",
                    "error_type": "http_error",
                    "execution_time": 0,
                    "return_code": -1,
                    "note": "Executor may have crashed or network connection is unstable. Execution may still be running on Trainium - check executor logs and CloudWatch metrics."
                }
        except requests.exceptions.RequestException as e:
            # Other request errors - don't retry
            logger.error(f"HTTP request to Trainium executor failed: {e}")
            return {
                "success": False,
                "error_message": f"HTTP request failed: {str(e)}",
                "error_type": "http_error",
                "execution_time": 0,
                "return_code": -1
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
    """Save code to a file in the appropriate results directory.
    Uses per-paper folder structure: results/{paper_id}/{step}/"""
    # Use per-paper folder structure: results/{paper_id}/{step}/
    paper_dir = RESULTS_DIR / paper_id
    save_dir = paper_dir / step
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{paper_id}_{timestamp}.py"
    filepath = save_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    
    logger.info(f"Saved code to {filepath}")
    return str(filepath)


def process_paper(paper_id: str, generator) -> Dict[str, Any]:
    """
    Process a single paper through the entire pipeline.
    
    Args:
        paper_id: OpenSearch document ID
        generator: PyTorchCodeGenerator instance
        
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
        
        # Step 3: Code review (only for standard generator, chunked doesn't do review yet)
        logger.info(f"Step 3: Code review results for paper {paper_id}...")
        step3_start = time.time()
        
        # Chunked generator doesn't do code review yet, standard generator does
        code_review = code_gen_result.get("code_review", {})
        reviewed_code = code_gen_result["code"]  # Already reviewed (or not, for chunked)
        
        # Save reviewed code
        reviewed_code_file = save_code_file(paper_id, reviewed_code, 'code-review')
        
        # Use actual review_time from code_review if available, otherwise measure pipeline step time
        actual_review_time = code_review.get("review_time", time.time() - step3_start)
        
        step3_result = {
            "success": True,
            "fixes_applied": code_review.get("fixes_applied", []),
            "iterations": code_review.get("iterations", 0),
            "reviewed_code_file": reviewed_code_file,
            "review_time": actual_review_time
        }
        
        save_step_result('code-review', paper_id, {
            "code": reviewed_code,
            "metadata": step3_result,
            "code_review_details": code_review
        })
        
        pipeline_results["steps"]["code_review"] = step3_result
        logger.info(f"✅ Code review complete ({step3_result['iterations']} iterations)")
        
        # Step 4: Execute on Trainium via HTTP
        if not TRAINIUM_ENDPOINT:
            logger.warning("⚠️ TRAINIUM_ENDPOINT not set - skipping execution")
            logger.warning("   Code generation and review completed successfully")
            pipeline_results["steps"]["trn_execution"] = {
                "success": False,
                "error": "TRAINIUM_ENDPOINT not set",
                "execution_time_seconds": 0,
                "skipped": True
            }
            pipeline_results["success"] = True  # Code generation succeeded
            return pipeline_results
        
        # Step 4a: Ensure Trainium is running
        logger.info(f"Step 4a: Ensuring Trainium instance is running...")
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
        
        # Step 4b: Execute on Trainium via HTTP
        logger.info(f"Step 4b: Executing code on Trainium for paper {paper_id}...")
        step4_start = time.time()
        
        # Use longer timeout to account for Neuron compilation (can take 20-40+ minutes)
        execution_timeout = int(os.getenv('TRAINIUM_EXECUTION_TIMEOUT', '3600'))  # 60 minutes default
        execution_result = execute_code_via_http(
            paper_id=paper_id,
            code=reviewed_code,
            timeout=execution_timeout,
            paper_title=code_gen_result.get("paper_title")
        )
        
        step4_result = {
            "success": execution_result.get("success", False),
            "execution_time": execution_result.get("execution_time", 0),
            "return_code": execution_result.get("return_code", -1),
            "timeout": execution_result.get("timeout", False),
            "metrics": execution_result.get("detailed_metrics", {}),
            "execution_time_seconds": time.time() - step4_start
        }
        
        # Save execution results
        save_step_result('trn-execution', paper_id, {
            "execution_result": execution_result,
            "metadata": step4_result,
            "stdout": execution_result.get("stdout", ""),
            "stderr": execution_result.get("stderr", "")
        })
        
        # Save metrics separately
        if step4_result.get("metrics"):
            save_step_result('metrics', paper_id, {
                "paper_id": paper_id,
                "paper_title": code_gen_result.get("paper_title"),
                "metrics": step4_result["metrics"],
                "execution_time": step4_result["execution_time"],
                "success": step4_result["success"]
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
        
        pipeline_results["steps"]["trn_execution"] = step4_result
        
        if execution_result.get("success"):
            logger.info(f"✅ Execution successful ({step4_result['execution_time']:.1f}s)")
        else:
            # Extract error message from stderr if available
            error_msg = execution_result.get('error_message')
            if not error_msg:
                stderr = execution_result.get('stderr', '')
                # Try to extract the actual error from stderr
                if stderr:
                    # Look for common error patterns
                    import re
                    # Find ValueError, RuntimeError, ModuleNotFoundError, etc.
                    error_match = re.search(r'(ValueError|RuntimeError|ModuleNotFoundError|ImportError|TypeError|AttributeError|KeyError|IndexError):\s*(.+)', stderr)
                    if error_match:
                        error_msg = f"{error_match.group(1)}: {error_match.group(2).split(chr(10))[0]}"
                    else:
                        # Fallback: get last non-empty line from stderr
                        stderr_lines = [line.strip() for line in stderr.split('\n') if line.strip()]
                        if stderr_lines:
                            error_msg = stderr_lines[-1][:200]  # Limit to 200 chars
                        else:
                            error_msg = f"Return code: {execution_result.get('return_code', -1)}"
                else:
                    error_msg = f"Return code: {execution_result.get('return_code', -1)}"
            
            logger.error(f"❌ Execution failed: {error_msg}")
        
        # Overall pipeline result
        pipeline_results["success"] = (
            step2_result.get("success") and 
            step3_result.get("success") and 
            step4_result.get("success")
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
    parser.add_argument('--use-chunked', action='store_true',
                       help='Use chunked code generator (for long papers that exceed token limits)')
    parser.add_argument('--chunked-parallel', action='store_true',
                       help='Process chunks in parallel (only with --use-chunked, may increase throttling risk)')
    parser.add_argument('--max-parallel', type=int, default=2,
                       help='Max parallel chunks (only with --use-chunked and --chunked-parallel, default: 2)')
    parser.add_argument('--enable-execution-testing', action='store_true',
                       help='Enable execution testing during code review (tests code on Trainium each iteration)')
    
    args = parser.parse_args()
    
    # Set environment variable for chunked generator if needed (must be before initialization)
    if args.enable_execution_testing:
        os.environ['ENABLE_EXECUTION_TESTING'] = 'true'
        logger.info("✅ Execution testing enabled - code will be tested on Trainium during review")
    
    # Initialize generator
    if args.use_chunked:
        if not CHUNKED_AVAILABLE:
            logger.error("❌ Chunked code generator requested but not available")
            logger.error("   Make sure chunked-code-gen module is properly set up")
            return
        logger.info("Initializing Chunked PyTorch Code Generator...")
        generator = ChunkedPyTorchGenerator(
            max_chunk_size=150000,  # 150k characters per chunk (accounts for prompt + paper summary overhead)
            use_haiku_for_chunks=True,
            parallel_chunks=args.chunked_parallel,
            max_parallel=args.max_parallel,
            batch_size=8  # Group 8 chunk summaries into each batch for hierarchical summarization
        )
        logger.info("✅ Using chunked approach (better for long papers)")
    else:
        logger.info("Initializing PyTorch Code Generator...")
        generator = PyTorchCodeGenerator(enable_execution_testing=args.enable_execution_testing)
        logger.info("✅ Using standard approach")
    
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

