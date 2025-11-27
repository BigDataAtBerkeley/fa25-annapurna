#!/usr/bin/env python3
"""
Trainium Executor Service v2

New architecture:
- /execute: Executes code, saves errors to DB on failure, calls /code_review
- /code_review: Reads errors from DB, fixes code, saves to S3, re-calls /execute
- Errors stored in database (not in-memory)
- Code stored in S3 bucket papers-code-artifacts
"""

from sys import stdout
from urllib.request import CacheFTPHandler
from flask import Flask, request, jsonify
import subprocess
import tempfile
import os
import time
import logging
import json
import traceback
import threading
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
import psutil
import shutil
import boto3
from botocore.exceptions import ClientError

# Import local modules
from error_db import save_error, get_errors, clear_errors
from s3_code_storage import save_code, get_code, code_exists

app = Flask(__name__)

# Configure logging
log_dir = os.path.join(os.path.expanduser('~'), 'trainium-executor', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'trainium-executor.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import SageMaker metrics logger
try:
    from sagemaker_metrics import SageMakerMetricsLogger, create_metrics_logger
    SAGEMAKER_METRICS_ENABLED = os.getenv('SAGEMAKER_METRICS_ENABLED', 'true').lower() == 'true'
except ImportError:
    SAGEMAKER_METRICS_ENABLED = False
    logger.warning("sagemaker_metrics module not found. Metrics logging to CloudWatch will be disabled.")

# Configuration
MAX_EXECUTION_TIME = int(os.getenv('MAX_EXECUTION_TIME', '1800'))  # 30 minutes
SUCCESS_ASSUMPTION_TIME = int(os.getenv('SUCCESS_ASSUMPTION_TIME', '300'))  # 5 minutes - assume success if running this long
WORKING_DIR = os.getenv('WORKING_DIR', '/tmp/trainium_jobs')
DATASET_CACHE_DIR = os.getenv('DATASET_CACHE_DIR', '/tmp/datasets')
NEURON_PROFILER_ENABLED = os.getenv('NEURON_PROFILER_ENABLED', 'true').lower() == 'true'
PROFILER_OUTPUT_DIR = os.getenv('PROFILER_OUTPUT_DIR', '/tmp/neuron_profiler')
RESULTS_BUCKET = os.getenv('RESULTS_BUCKET', 'trainium-execution-results')
MAX_REVIEW_ITERATIONS = int(os.getenv('MAX_REVIEW_ITERATIONS', '5'))  # Max code review iterations

# DynamoDB Error Database Configuration
# Set ERROR_DB_TABLE_NAME environment variable with your DynamoDB table name
# Table structure: Partition key = DOC#<docId>, Sort key = ITER#<iteration>#ERR#<errorId>
ERROR_DB_TABLE_NAME = os.getenv('ERROR_DB_TABLE_NAME', 'docRunErrors')

# Bedrock configuration for code review
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')

# Initialize Bedrock client for code review
bedrock_client = None
try:
    from botocore.config import Config
    config = Config(read_timeout=150, retries={'max_attempts': 0})
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=config)
    logger.info(f"Bedrock client initialized for code review: {BEDROCK_MODEL_ID}")
except Exception as e:
    logger.error(f"Failed to initialize Bedrock client: {e}")

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
os.makedirs(PROFILER_OUTPUT_DIR, exist_ok=True)

# Track running executions (for 5-minute success assumption)
running_executions = {}
execution_lock = threading.Lock()

def collect_profiler_artifacts(paper_id: str, profiler_path: str):
    if not os.path.exists(profiler_path):
        return
    
    try:
        s3_client = boto3.client('s3')
        for root, dirs, files in os.walk(profiler_path):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = f"profiler/{paper_id}/{file}"
                s3_client.upload_file(local_path, RESULTS_BUCKET, s3_key)
        logger.info(f"Uploaded profiler artifacts for {paper_id}")
    except Exception as e:
        logger.error(f"Failed to upload profiler artifacts: {e}")

def upload_execution_results(paper_id: str, result: Dict[str, Any]):
    try:
        s3_client = boto3.client('s3')
        s3_key = f"results/{paper_id}/execution_result.json"
        s3_client.put_object(
            Bucket=RESULTS_BUCKET,
            Key=s3_key,
            Body=json.dumps(result, indent=2)
        )
        logger.info(f"Uploaded execution results for {paper_id}")
    except Exception as e:
        logger.error(f"Failed to upload results: {e}")

def execute_code_sync(paper_id: str, code: str, timeout: int, paper_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute code synchronously (blocking).
    This is the core execution function used by both /execute and /code_review.
    
    Args:
        paper_id: Paper/document ID
        code: Python code to execute
        timeout: Maximum execution time
        paper_title: Paper title (optional)
        
    Returns:
        Execution result dictionary
    """
    exec_dir = tempfile.mkdtemp(prefix=f'trainium_exec_{paper_id}_', dir='/tmp')
    
    try:
        # Write main.py
        main_py_path = os.path.join(exec_dir, 'main.py')
        with open(main_py_path, 'w') as f:
            f.write(f"""
import sys
import os
sys.path.append('{exec_dir}')

{code}
""")
        
        # Copy dataset_loader.py
        loader_source = os.path.join(os.path.dirname(__file__), 'dataset_loader.py')
        loader_dest = os.path.join(exec_dir, 'dataset_loader.py')
        if os.path.exists(loader_source):
            shutil.copy2(loader_source, loader_dest)
        
        # Remove any torch conflicts
        for item in os.listdir(exec_dir):
            item_path = os.path.join(exec_dir, item)
            if item.lower() in ['torch', '_torch', 'torch.py', '_torch.py']:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # Initialize metrics logger
        metrics_logger = None
        if SAGEMAKER_METRICS_ENABLED:
            try:
                instance_type = os.getenv('TRAINIUM_INSTANCE_TYPE', 'trn1.2xlarge')
                metrics_logger = create_metrics_logger(
                    paper_id=paper_id,
                    paper_title=paper_title,
                    instance_type=instance_type
                )
            except Exception as e:
                logger.warning(f"Failed to initialize metrics logger: {e}")
        
        # Set up Neuron environment
        user_home = os.path.expanduser('~')
        neuron_bin_paths = [
            '/opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin',
            '/opt/aws_neuronx_venv_pytorch_2_8/bin',
            f'{user_home}/.local/bin',
            '/opt/aws/neuron/bin',
            '/usr/local/bin',
            '/usr/bin',
            '/bin'
        ]
        current_path = os.environ.get('PATH', '')
        neuron_path = ':'.join(neuron_bin_paths + [current_path])
        neuron_python = '/opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin/python3'
        
        # Set up profiler
        profiler_output_path = None
        if NEURON_PROFILER_ENABLED:
            profiler_output_path = os.path.join(PROFILER_OUTPUT_DIR, f'{paper_id}_{int(time.time())}')
            os.makedirs(profiler_output_path, exist_ok=True)
        
        exec_env = {
            **os.environ,
            'PATH': neuron_path,
            'PYTHONUNBUFFERED': '1',
            'DATASET_CACHE_DIR': DATASET_CACHE_DIR,
            'PYTHONDONTWRITEBYTECODE': '1',
            'NEURON_RT_LOG_LEVEL': 'ERROR'
        }
        
        # Build command
        if NEURON_PROFILER_ENABLED and profiler_output_path:
            neuron_profile_cmd = '/opt/aws/neuron/bin/neuron-profile'
            if os.path.exists(neuron_profile_cmd):
                cmd = [
                    neuron_profile_cmd,
                    'inspect',
                    '-o', profiler_output_path,
                    neuron_python, main_py_path
                ]
            else:
                cmd = [neuron_python, main_py_path]
        else:
            cmd = [neuron_python, main_py_path]
        
        # Track execution start
        start_time = time.time()
        with execution_lock:
            running_executions[paper_id] = {
                'start_time': start_time,
                'paper_id': paper_id
            }
        
        logger.info(f"Executing code for {paper_id} (timeout: {timeout}s)")
        
        # Execute
        result = subprocess.run(
            cmd,
            cwd='/tmp',
            capture_output=True,
            text=True,
            timeout=timeout,
            env=exec_env
        )
        
        execution_time = time.time() - start_time
        
        # Remove from running executions
        with execution_lock:
            running_executions.pop(paper_id, None)
        
        success = result.returncode == 0
        
        # Extract metrics
        metrics = {}
        try:
            for line in result.stdout.split('\n'):
                if line.startswith('METRICS:'):
                    json_str = line.replace('METRICS:', '').strip()
                    metrics.update(json.loads(json_str))
        except:
            pass
        
        # Log metrics
        if metrics_logger:
            try:
                metrics_logger.extract_and_log_metrics_from_output(result.stdout)
                metrics_logger.log_execution_metrics(
                    execution_time=execution_time,
                    success=success,
                    peak_memory_mb=0
                )
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")
        
        execution_result = {
            "success": success,
            "execution_time": round(execution_time, 2),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timeout": False,
            "profiler_output_path": profiler_output_path,
            "detailed_metrics": metrics,
            **metrics
        }
        
        return execution_result
        
    except subprocess.TimeoutExpired as e:
        execution_time = time.time() - start_time
        with execution_lock:
            running_executions.pop(paper_id, None)
        
        return {
            "success": False,
            "execution_time": timeout,
            "return_code": -1,
            "stdout": e.stdout.decode('utf-8') if e.stdout else "",
            "stderr": e.stderr.decode('utf-8') if e.stderr else "",
            "timeout": True,
            "error_message": f"Execution timed out after {timeout} seconds",
            "error_type": "timeout"
        }
        
    except Exception as e:
        with execution_lock:
            running_executions.pop(paper_id, None)
        
        logger.error(f"Error executing code for {paper_id}: {e}")
        return {
            "success": False,
            "execution_time": 0,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "error_message": f"Execution error: {str(e)}",
            "error_type": "execution_error"
        }
    
    finally:
        # Cleanup
        try:
            if os.path.exists(exec_dir):
                shutil.rmtree(exec_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup {exec_dir}: {e}")
            

def extract_errors_from_result(exec_result: Dict[str, Any]) -> str:
    
    stderr = exec_result.get('stderr', '')
    message = exec_result.get('error_message', '')
    stdout = exec_result.get('stdout', '')
    
    if not (stderr or message or stdout):
        return ''
    
    if exec_result.get('timeout') or exec_result.get('error_type') == 'timeout':
        return f"EXECUTION ERROR: Execution timed out - {exec_result.get('error_message', 'Timeout')}"
    
    error_message = (
        f"Error Message: {message if message else 'Code execution failed.'}. "
        f"Standard Error: {stderr if stderr else -1}"
    )
    
    return error_message


def fix_code_with_bedrock(code: str, error_message: str, iteration: int) -> Optional[str]:

    if not bedrock_client:
        logger.error("Bedrock client not available for code fixing")
        return None
    
    prompt = f"""You are fixing PyTorch code that failed execution on AWS Trainium.

The code failed with these REAL execution errors:
{error_message}

Current code (iteration {iteration}):
```python
{code}
```

CRITICAL REQUIREMENTS:
1. Fix ALL the errors listed above
2. Code MUST use torch_xla (Neuron SDK) for Trainium
3. Use xm.xla_device(), xm.optimizer_step(), xm.mark_step()
4. Do NOT use non-existent xm APIs (xm.XlaModule, xm.dot_general, etc.)
5. Ensure all imports are present
6. Fix any syntax errors, type errors, attribute errors

Return ONLY the complete fixed code in a Python code block. Do not include explanations outside the code block.

Fixed code:
```python
"""
    
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        generated_text = response_body['content'][0]['text']
        
        # Extract code from response
        import re
        code_match = re.search(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
        if code_match:
            fixed_code = code_match.group(1)
            logger.info(f"Fixed code extracted (length: {len(fixed_code)})")
            return fixed_code
        else:
            # Try to extract without markdown
            fixed_code = generated_text.strip()
            if fixed_code.startswith('import') or fixed_code.startswith('from'):
                logger.info(f"Fixed code extracted (no markdown, length: {len(fixed_code)})")
                return fixed_code
        
        logger.warning("Failed to extract code from Bedrock response")
        return None
        
    except Exception as e:
        logger.error(f"Failed to fix code with Bedrock: {e}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "neuron_available": os.path.exists('/opt/aws/neuron'),
        "bedrock_available": bedrock_client is not None
    })

@app.route('/env', methods=['GET'])
def get_env_vars():
    """Print all environment variables"""
    return jsonify({
        "environment_variables": dict(os.environ)
    })


def execute_internal(paper_id: str, code: str, timeout: int, paper_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Internal execute function (can be called directly or via HTTP endpoint).
    Returns dictionary instead of Flask response.
    """
    logger.info(f"Executing code for {paper_id}")
    
    # Save code to S3 first
    s3_key = save_code(paper_id, code)
    if not s3_key:
        logger.warning(f"Failed to save code to S3 for {paper_id}, continuing anyway")
    
    # Execute code in background thread
    def execute_and_handle():
        exec_result = execute_code_sync(paper_id, code, timeout, paper_title)
        
        if exec_result.get('success'):
            logger.info(f"Execution succeeded for {paper_id}")
            upload_execution_results(paper_id, exec_result)
            profiler_trace = "profiler_output_path"
            if profiler_trace:
                collect_profiler_artifacts(paper_id, exec_result.get("profiler_output_path"))
            # Success - clear any previous errors
            clear_errors(paper_id)
        else:
            # Failure - save error to DB and trigger code review
            logger.warning(f"Execution failed for {paper_id}, saving error and triggering review")
            error_data = {
                "stderr": exec_result.get('stderr', ''),
                "stdout": exec_result.get('stdout', ''),
                "error_message": exec_result.get('error_message', ''),
                "error_type": exec_result.get('error_type', 'execution_error'),
                "return_code": exec_result.get('return_code', -1),
                "execution_time": exec_result.get('execution_time', 0)
            }
            save_error(paper_id, error_data)
            
            # Call code_review endpoint
            try:
                review_response = requests.post(
                    f"http://localhost:8000/code_review",
                    json={"paper_id": paper_id, "paper_title": paper_title},
                    timeout=300  # 5 minute timeout for review
                )
                logger.info(f"Code review triggered for {paper_id}: {review_response.status_code}")
            except Exception as e:
                logger.error(f"Failed to trigger code review: {e}")
    
    # Start execution in background
    thread = threading.Thread(target=execute_and_handle, daemon=True)
    thread.start()
    
    # Check if execution is still running after SUCCESS_ASSUMPTION_TIME
    def check_success_assumption():
        time.sleep(SUCCESS_ASSUMPTION_TIME)
        with execution_lock:
            if paper_id in running_executions:
                # Still running after 5 minutes - assume success
                logger.info(f"Execution for {paper_id} running > {SUCCESS_ASSUMPTION_TIME}s, assuming success")
    
    # Start success assumption check in background
    check_thread = threading.Thread(target=check_success_assumption, daemon=True)
    check_thread.start()
    
    # Return status
    return {
        "success": True,
        "status": "running",
        "job_id": paper_id,
        "status_url": f"/status/{paper_id}",
        "message": f"Execution started. If running > {SUCCESS_ASSUMPTION_TIME}s, assume success. ECS should track via /status endpoint."
    }


@app.route('/execute', methods=['POST'])
def execute():
    """
    Execute code on Trainium.
    
    Request body:
    {
        "paper_id": "paper_123",
        "code": "import torch\n...",
        "timeout": 1800,
        "paper_title": "Paper Title"
    }
    
    Behavior:
    - Executes code
    - If fails: saves error to DB, calls /code_review
    - If runs > 5 minutes: assumes success, returns job_id for ECS to track
    - If succeeds quickly: returns results immediately
    """
    try:
        data = request.get_json()
        paper_id = data.get('paper_id')
        code = data.get('code')
        timeout = data.get('timeout', MAX_EXECUTION_TIME)
        paper_title = data.get('paper_title')
        logger.info(f"Recieved code for {paper_id}, Title: {paper_title}")
        
        if not paper_id or not code:
            return jsonify({
                "success": False,
                "error": "paper_id and code are required"
            }), 400
        
        # Call internal execute function
        logger.info(f"Beginning internal execution.")
        result = execute_internal(paper_id, code, timeout, paper_title)
        return jsonify(result), 202
        
    except Exception as e:
        logger.error(f"Error in /execute: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/code_review', methods=['POST'])
def code_review():
    """
    Review and fix code based on errors in database.
    
    Request body:
    {
        "paper_id": "paper_123"
    }
    
    Behavior:
    - Reads errors from database for paper_id
    - Gets current code from S3
    - Uses Bedrock to fix code
    - Saves fixed code to S3 (replaces old)
    - Re-calls /execute with fixed code
    """
    try:
        data = request.get_json()
        paper_id = data.get('paper_id')
        
        if not paper_id:
            return jsonify({
                "success": False,
                "error": "paper_id is required"
            }), 400
        
        logger.info(f"Starting code review for {paper_id}")
        
        errors_list = get_errors(paper_id)
        
        error_count = len(errors_list)
        logger.info(f"Retrieved {error_count} previous iterations for this paper. Passing in the most recent.")
        
        if error_count >= MAX_REVIEW_ITERATIONS:
            logger.warning(f"Max code review depth reached for {paper_id} ({error_count})")
            return jsonify({
                "success": False,
                "error": f"Max review iterations ({MAX_REVIEW_ITERATIONS}) reached",
                "paper_id": paper_id,
                "error_count": error_count
            }), 400
            
        # Possibly explore providing context including more than just the most recent error?   
        latest_error = errors_list[-1].get('error_data', {})
        error_message = extract_errors_from_result(latest_error)
        if not error_message:
            return jsonify({
                "success": False,
                "error": "No extractable errors found"
            }), 400

        
        # Get current code from S3
        code = get_code(paper_id)
        if not code:
            return jsonify({
                "success": False,
                "error": f"No code found in S3 for {paper_id}"
            }), 404
        
        # Fix code with Bedrock
        logger.info(f"Fixing code with Bedrock (iteration {error_count})...")
        fixed_code = fix_code_with_bedrock(code, error_message, error_count)
        
        if not fixed_code:
            return jsonify({
                "success": False,
                "error": "Failed to fix code with Bedrock"
            }), 500
        
        # Save fixed code to S3 (replaces old)
        s3_key = save_code(paper_id, fixed_code)
        if not s3_key:
            return jsonify({
                "success": False,
                "error": "Failed to save fixed code to S3"
            }), 500
        
        logger.info(f"Fixed code saved to S3: {s3_key}")
        
        # Re-call /execute with fixed code (recursive call)
        logger.info(f"Re-executing fixed code for {paper_id}...")
        try:
            # Call internal execute function directly (not HTTP)
            exec_result = execute_internal(
                paper_id=paper_id,
                code=fixed_code,
                timeout=MAX_EXECUTION_TIME,
                paper_title=data.get('paper_title')
            )
            
            return jsonify({
                "success": True,
                "message": f"Code fixed and re-execution triggered (iteration {error_count})",
                "paper_id": paper_id,
                "iteration": error_count,
                "errors_fixed": error_message,
                "execution_status": exec_result
            })
                
        except Exception as e:
            logger.error(f"Failed to re-execute code: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "success": False,
                "error": f"Failed to re-execute: {str(e)}",
                "paper_id": paper_id
            }), 500
        
    except Exception as e:
        logger.error(f"Error in /code_review: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/status/<paper_id>', methods=['GET'])
def get_status(paper_id: str):
    """Get execution status for a paper_id."""
    with execution_lock:
        if paper_id in running_executions:
            exec_info = running_executions[paper_id]
            elapsed = time.time() - exec_info['start_time']
            return jsonify({
                "paper_id": paper_id,
                "status": "running",
                "elapsed_seconds": round(elapsed, 2),
                "assumed_success": elapsed > SUCCESS_ASSUMPTION_TIME
            })
    
    # Check if code exists in S3 (indicates execution completed)
    if code_exists(paper_id):
        errors = get_errors(paper_id)
        if errors:
            return jsonify({
                "paper_id": paper_id,
                "status": "failed",
                "error_count": len(errors),
                "latest_error": errors[-1].get('error_data', {})
            })
        else:
            return jsonify({
                "paper_id": paper_id,
                "status": "completed",
                "success": True
            })
    
    return jsonify({
        "paper_id": paper_id,
        "status": "not_found"
    }), 404


if __name__ == '__main__':
    logger.info("Starting Trainium Executor Service v2")
    logger.info(f"Working directory: {WORKING_DIR}")
    logger.info(f"Max execution time: {MAX_EXECUTION_TIME}s")
    logger.info(f"Success assumption time: {SUCCESS_ASSUMPTION_TIME}s")
    logger.info(f"Code storage bucket: papers-code-artifacts")
    logger.info(f"Max review iterations: {MAX_REVIEW_ITERATIONS}")
    
    # Print all environment variables
    logger.info("=" * 80)
    logger.info("Environment Variables:")
    logger.info("=" * 80)
    for key, value in sorted(os.environ.items()):
        logger.info(f"{key}={value}")
    logger.info("=" * 80)
    
    app.run(host='0.0.0.0', port=8000, threaded=True)

