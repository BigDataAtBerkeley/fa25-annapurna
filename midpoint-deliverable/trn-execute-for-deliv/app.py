#!/usr/bin/env python3
"""
Trainium Executor Service v2

New architecture:
- /execute: Executes code, saves errors to DB on failure, calls /code_review
- /code_review: Reads errors from DB, fixes code, saves to S3, re-calls /execute
- Errors stored in database (not in-memory)
- Code stored in S3 bucket papers-code-artifacts
"""

import sys
import os

# Add script directory to Python path so we can import local modules
# This is critical when running via systemd where the working directory might differ
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

code_gen_dir = os.path.join(os.path.dirname(script_dir), 'code-gen-for-deliv')
home_code_gen = os.path.join(os.path.expanduser('~'), 'code-gen-for-deliv')

code_gen_dir_used = None
for path in [code_gen_dir, home_code_gen]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        code_gen_dir_used = path
        break

if code_gen_dir_used:
    code_gen_dir = code_gen_dir_used
elif os.path.exists(home_code_gen):
    code_gen_dir = home_code_gen

from flask import Flask, request, jsonify
import subprocess
import tempfile
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
from error_db import save_error, get_errors, get_all_errors, clear_errors
from s3_code_storage import save_code, get_code, code_exists

OPENSEARCH_AVAILABLE = False
SLACK_AVAILABLE = False
OpenSearchClient = None
SlackNotifier = None

try:
    # Try importing with explicit path check
    import importlib.util
    opensearch_path = os.path.join(code_gen_dir, 'opensearch_client.py')
    slack_path = os.path.join(code_gen_dir, 'slack_notifier.py')
    
    if not os.path.exists(opensearch_path):
        opensearch_path = os.path.join(home_code_gen, 'opensearch_client.py')
        slack_path = os.path.join(home_code_gen, 'slack_notifier.py')
    
    if os.path.exists(opensearch_path) and os.path.exists(slack_path):
        # Load modules explicitly
        spec1 = importlib.util.spec_from_file_location("opensearch_client", opensearch_path)
        opensearch_module = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(opensearch_module)
        OpenSearchClient = opensearch_module.OpenSearchClient
        
        spec2 = importlib.util.spec_from_file_location("slack_notifier", slack_path)
        slack_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(slack_module)
        SlackNotifier = slack_module.SlackNotifier
        
        OPENSEARCH_AVAILABLE = True
        SLACK_AVAILABLE = True
    else:
        # Fallback to regular import
        from opensearch_client import OpenSearchClient
        from slack_notifier import SlackNotifier
        OPENSEARCH_AVAILABLE = True
        SLACK_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: OpenSearch/Slack modules not available: {e}. Slack notifications will be disabled.", file=sys.stderr)
    print(f"WARNING: Code gen directory: {code_gen_dir}", file=sys.stderr)
    print(f"WARNING: Home code gen: {home_code_gen}", file=sys.stderr)
    print(f"WARNING: Code gen directory exists: {os.path.exists(code_gen_dir)}", file=sys.stderr)
    print(f"WARNING: Home code gen exists: {os.path.exists(home_code_gen)}", file=sys.stderr)
    if os.path.exists(code_gen_dir):
        print(f"WARNING: Files in code-gen-for-deliv: {os.listdir(code_gen_dir)[:10]}", file=sys.stderr)
    if os.path.exists(home_code_gen):
        print(f"WARNING: Files in home code-gen-for-deliv: {os.listdir(home_code_gen)[:10]}", file=sys.stderr)
except Exception as e:
    print(f"WARNING: Error loading OpenSearch/Slack modules: {e}. Slack notifications will be disabled.", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    if not OPENSEARCH_AVAILABLE:
        OPENSEARCH_AVAILABLE = False
    if not SLACK_AVAILABLE:
        SLACK_AVAILABLE = False

app = Flask(__name__)

# Configure logging with fallback if log directory can't be created
log_dir = os.path.join(os.path.expanduser('~'), 'trainium-executor', 'logs')
log_file = os.path.join(log_dir, 'trainium-executor.log')

# Try to create log directory, but don't fail if it doesn't work
try:
    os.makedirs(log_dir, exist_ok=True)
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
except Exception as e:
    # If we can't create log directory, just use console logging
    print(f"WARNING: Could not create log directory {log_dir}: {e}", file=sys.stderr)
    print("WARNING: Logging to console only", file=sys.stderr)
    handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Log OpenSearch/Slack availability after logger is initialized
if OPENSEARCH_AVAILABLE and SLACK_AVAILABLE:
    logger.info("✅ OpenSearch and Slack modules imported successfully")
else:
    logger.warning(f"⚠️ OpenSearch/Slack modules not available - Slack notifications will be disabled")
    logger.warning(f"   OPENSEARCH_AVAILABLE: {OPENSEARCH_AVAILABLE}, SLACK_AVAILABLE: {SLACK_AVAILABLE}")
    logger.warning(f"   Code gen directory: {code_gen_dir}")
    logger.warning(f"   Code gen exists: {os.path.exists(code_gen_dir)}")
    if os.path.exists(code_gen_dir):
        py_files = [f for f in os.listdir(code_gen_dir) if f.endswith('.py')]
        logger.warning(f"   Code gen Python files: {', '.join(py_files[:5])}")

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
MAX_REVIEW_ITERATIONS = int(os.getenv('MAX_REVIEW_ITERATIONS', '50'))  # Increased limit for new strategy
CODE_REVIEW_STABILITY_TIME = int(os.getenv('CODE_REVIEW_STABILITY_TIME', '120'))  # 2 minutes - if code runs this long without errors, consider it stable

# DynamoDB Error Database Configuration
# Set ERROR_DB_TABLE_NAME environment variable with your DynamoDB table name
# Table structure: Partition key = DOC#<docId>, Sort key = ITER#<iteration>#ERR#<errorId>
ERROR_DB_TABLE_NAME = os.getenv('ERROR_DB_TABLE_NAME', 'docRunErrors')

# Bedrock configuration for code review
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
# Using Claude 3 Sonnet (supports on-demand, widely available, not legacy)
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

def send_slack_notification(paper_id: str, execution_result: Dict[str, Any]):
    """
    Send Slack notification with paper info, execution results, and code links.
    Excludes embeddings and other large binary fields.
    
    Args:
        paper_id: Paper/document ID
        execution_result: Execution result dictionary
    """
    if not SLACK_AVAILABLE or not OPENSEARCH_AVAILABLE:
        logger.warning(f"⚠️ Slack/OpenSearch not available - skipping notification for {paper_id}")
        logger.warning(f"   SLACK_AVAILABLE={SLACK_AVAILABLE}, OPENSEARCH_AVAILABLE={OPENSEARCH_AVAILABLE}")
        logger.warning(f"   To enable Slack notifications, ensure opensearch_client and slack_notifier modules are available")
        return
    
    try:
        # Initialize clients
        opensearch_client = OpenSearchClient()
        slack_notifier = SlackNotifier()
        
        # Get paper from OpenSearch
        try:
            paper = opensearch_client.get_paper_by_id(paper_id)
            if not paper:
                logger.warning(f"Paper {paper_id} not found in OpenSearch (index: {opensearch_client.opensearch_index}), skipping Slack notification")
                return
        except Exception as e:
            logger.error(f"Error retrieving paper {paper_id} from OpenSearch: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Filter out embeddings and other large binary fields
        fields_to_exclude = {
            'embedding', 'embeddings', 'vector', 'vectors', 
            'pdf_bytes', 'pdf_content', 'raw_content',
            's3_bucket', 's3_key'  # We'll add formatted S3 link instead
        }
        
        # Create filtered paper dict with key fields
        filtered_paper = {
            '_id': paper_id,
            'title': paper.get('title', 'Unknown Title'),
            'authors': paper.get('authors', []),
            'abstract': paper.get('abstract', ''),
            'date': paper.get('date', ''),
            'venue': paper.get('venue', ''),
            'url': paper.get('url', ''),
            'arxiv_id': paper.get('arxiv_id', ''),
        }
        
        # Add other non-excluded fields
        for key, value in paper.items():
            if key not in fields_to_exclude and key not in filtered_paper:
                # Skip very large fields
                if isinstance(value, str) and len(value) > 2000:
                    continue
                if isinstance(value, (dict, list)) and len(str(value)) > 2000:
                    continue
                filtered_paper[key] = value
        
        # Add execution result info
        filtered_paper['execution_success'] = execution_result.get('success', False)
        filtered_paper['execution_time_seconds'] = round(execution_result.get('execution_time', 0), 2)
        filtered_paper['execution_return_code'] = execution_result.get('return_code', -1)
        
        # Add execution details
        if execution_result.get('success'):
            filtered_paper['execution_status'] = '✅ SUCCESS'
            if execution_result.get('stdout'):
                stdout_preview = execution_result.get('stdout', '')[:1000]
                filtered_paper['execution_stdout_preview'] = stdout_preview + ('...' if len(execution_result.get('stdout', '')) > 1000 else '')
        else:
            filtered_paper['execution_status'] = '❌ FAILED'
            filtered_paper['execution_error'] = execution_result.get('error_message', 'Unknown error')
            if execution_result.get('stderr'):
                stderr_preview = execution_result.get('stderr', '')[:1000]
                filtered_paper['execution_stderr_preview'] = stderr_preview + ('...' if len(execution_result.get('stderr', '')) > 1000 else '')
        
        # Add S3 code link
        code_s3_key = f"code/{paper_id}.py"
        filtered_paper['code_s3_location'] = f"s3://papers-code-artifacts/{code_s3_key}"
        filtered_paper['code_s3_url'] = f"https://s3.console.aws.amazon.com/s3/object/papers-code-artifacts?prefix={code_s3_key}"
        
        # Add execution results S3 link if available
        if execution_result.get('s3_results_key'):
            filtered_paper['results_s3_location'] = execution_result.get('s3_results_key')
        
        # Send to Slack using custom formatting
        success = slack_notifier.send_execution_notification(filtered_paper, execution_result)
        if success:
            logger.info(f"✅ Sent Slack notification for {paper_id}")
        else:
            logger.warning(f"Failed to send Slack notification for {paper_id}")
            
    except Exception as e:
        logger.error(f"Error sending Slack notification for {paper_id}: {e}")
        logger.error(traceback.format_exc())


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
        
        # Check return code first
        success = result.returncode == 0
        
        # Check stderr for critical errors even if return code is 0
        # Some errors (like Neuron internal errors) may not set non-zero return code
        stderr_lower = result.stderr.lower() if result.stderr else ""
        critical_errors = [
            "internal error",
            "fatal error",
            "segmentation fault",
            "core dumped",
            "abort",
            "assertion failed",
            "improper teardown",
            "object(s) leaked"
        ]
        
        has_critical_error = any(error in stderr_lower for error in critical_errors)
        if has_critical_error:
            logger.warning(f"Critical error detected in stderr despite return_code={result.returncode}")
            success = False
        
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
        
        # Add error information if critical error detected
        if has_critical_error and result.returncode == 0:
            # Extract the critical error message
            error_lines = [line for line in result.stderr.split('\n') 
                          if any(err in line.lower() for err in critical_errors)]
            if error_lines:
                execution_result["error_message"] = error_lines[-1].strip()
                execution_result["error_type"] = "critical_runtime_error"
                logger.error(f"Critical error in execution: {execution_result['error_message']}")
        
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


def fix_code_with_bedrock(code: str, error_message: str, iteration: int, paper_summary: Optional[str] = None, error_history: Optional[str] = None, all_past_errors: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:

    if not bedrock_client:
        logger.error("Bedrock client not available for code fixing")
        return None
    
    # Build paper context section
    paper_context = ""
    if paper_summary:
        paper_context = f"""
PAPER CONTEXT (for better understanding of what the code should implement):
{paper_summary}

"""
    
    # Build error history section
    history_context = ""
    if error_history:
        history_context = f"""
═══════════════════════════════════════════════════════════════════════════════
PREVIOUS ERRORS (from earlier code review iterations for THIS paper):
═══════════════════════════════════════════════════════════════════════════════
{error_history}

IMPORTANT: The errors above were from previous fix attempts for this paper. The fixes applied did not resolve these issues. Make sure your fix addresses BOTH the previous errors AND the current error below.

═══════════════════════════════════════════════════════════════════════════════
CURRENT ERROR (most recent execution):
═══════════════════════════════════════════════════════════════════════════════
"""
    
    # Build past errors from all papers section
    past_errors_context = ""
    if all_past_errors:
        # Extract unique error patterns from all past errors
        unique_errors = []
        seen_patterns = set()
        
        for past_error in all_past_errors[:50]:  # Limit to 50 most recent for prompt size
            error_data = past_error.get('error_data', {})
            error_msg = error_data.get('error_message', '') or error_data.get('stderr', '')[:200]
            
            # Extract key error patterns (first line of error message)
            if error_msg:
                first_line = error_msg.split('\n')[0].strip()
                # Create a pattern key to avoid duplicates
                pattern_key = first_line[:100]  # Use first 100 chars as pattern
                
                if pattern_key and pattern_key not in seen_patterns:
                    seen_patterns.add(pattern_key)
                    paper_id = past_error.get('paper_id', 'unknown')
                    unique_errors.append(f"  - {first_line[:150]} (from paper: {paper_id})")
        
        if unique_errors:
            past_errors_context = f"""
═══════════════════════════════════════════════════════════════════════════════
PAST ERRORS FROM ALL PAPERS (proactive checking):
═══════════════════════════════════════════════════════════════════════════════
The following errors occurred in OTHER papers when running on Trainium. 
Check your code to ensure these errors will NOT occur:

{chr(10).join(unique_errors[:30])}  # Limit to 30 for prompt size

CRITICAL: Review your code carefully and ensure it does NOT contain patterns that would cause these errors. This is proactive error prevention.

═══════════════════════════════════════════════════════════════════════════════
"""
    
    prompt = f"""You are fixing PyTorch code that failed execution on AWS Trainium.

{paper_context}{past_errors_context}{history_context}The code failed with these REAL execution errors:
{error_message}

Current code (iteration {iteration}):
```python
{code}
```

═══════════════════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS - READ CAREFULLY
═══════════════════════════════════════════════════════════════════════════════

1. **FIX ALL ERRORS**: Address EVERY error listed above. These are REAL runtime errors from Trainium execution.

2. **AWS NEURON SDK (MANDATORY - CRITICAL)**:
   - MUST use `torch_xla.core.xla_model as xm` - this is the Neuron SDK XLA interface
   - MUST use `device = xm.xla_device()` to get Trainium device (NOT torch.device('cuda') or 'cpu')
   - MUST use `xm.optimizer_step(optimizer)` instead of `optimizer.step()` - this is Neuron SDK requirement
   - MUST call `xm.mark_step()` after each backward pass to synchronize XLA computation
   - MUST move ALL tensors to device BEFORE any operations: `inputs = inputs.to(device)`, `labels = labels.to(device)`
   - DO NOT move loss functions: `criterion = nn.CrossEntropyLoss()` (no .to(device))
   - DO NOT use CUDA: no `.cuda()`, no `device='cuda'`, no `torch.device('cuda')`

3. **COMMON TRAINIUM ERRORS TO FIX**:
   - **Import errors**: Ensure ALL imports are present (math, random, collections, etc.)
   - **XLA tensor size**: `tensor.size(0)` returns a tensor in XLA, use `int(tensor.size(0))` for arithmetic
   - **Model output tuples**: Check `isinstance(model_output, tuple)` and unpack correctly
   - **Dataset tokenization**: IMDB returns text strings - MUST tokenize before use
   - **Shape mismatches**: LayerNorm/BatchNorm expect specific shapes - ensure input shape matches normalized_shape
   - **LoRA errors**: If using LoRA, ensure `B @ A` produces `[out_features, in_features]` (NOT `B @ A.T`)
   - **Parameter reassignment**: NEVER reassign `layer.weight = ...` in forward() - use `F.linear()` instead
   - **Ellipsis placeholders**: Replace any `variable = ...` with actual initialization
   - **ModuleDict keys**: Keys cannot contain dots - replace with underscores

4. **DO NOT USE NON-EXISTENT APIs**:
   - ❌ `xm.optimizer.SGD` - DOES NOT EXIST (use `torch.optim.SGD` then `xm.optimizer_step(optimizer)`)
   - ❌ `xm.XlaModule` - DOES NOT EXIST (use regular `nn.Module`)
   - ❌ `xm.dot()` or `xm.dot_general()` - DO NOT EXIST (use `torch.matmul()` or `torch.mm()`)
   - ❌ `xm.tensor()` - DOES NOT EXIST (use `torch.tensor()`)
   - ❌ `xm.xla_device_context()` - DOES NOT EXIST (use `device = xm.xla_device()` and `model.to(device)`)
   - ❌ `xm.optimizer_step(optimizer, sync=True)` - sync parameter DOES NOT EXIST

5. **VALID torch_xla APIs ONLY**:
   - ✅ `xm.xla_device()` - Get XLA device
   - ✅ `xm.optimizer_step(optimizer)` - XLA optimizer step (NO sync parameter)
   - ✅ `xm.mark_step()` - Synchronize XLA computation
   - ✅ `xm.rendezvous(tag)` - Synchronization barrier (requires tag string)

6. **DATA HANDLING**:
   - MUST use `from dataset_loader import load_dataset` - DO NOT use torchvision.datasets
   - Move tensors to device INSIDE training loop: `inputs = inputs.to(device)`, NOT the DataLoader
   - IMDB dataset: Returns (text_strings, labels) - MUST tokenize text strings before use

7. **CODE COMPLETENESS**:
   - Ensure all functions/classes have complete bodies (no ellipsis `...` placeholders)
   - All variables must be initialized with actual values, not placeholders
   - Ensure proper error handling and type checking

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
        
        # Send Slack notification after execution completes (success or failure)
        logger.info(f"Execution completed for {paper_id}, sending Slack notification...")
        try:
            send_slack_notification(paper_id, exec_result)
            logger.info(f"Slack notification attempt completed for {paper_id}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification for {paper_id}: {e}")
            logger.error(traceback.format_exc())
        
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
            
            # Call code_review endpoint with error_data as fallback
            try:
                review_payload = {
                    "paper_id": paper_id, 
                    "paper_title": paper_title,
                    "error_data": error_data  # Pass error directly in case DynamoDB fails
                }
                review_response = requests.post(
                    f"http://localhost:8000/code_review",
                    json=review_payload,
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
    
    NEW STRATEGY: Keep running until code executes for 2 minutes without errors.
    This replaces the fixed 4-iteration limit.
    
    Request body:
    {
        "paper_id": "paper_123"
    }
    
    Behavior:
    - Reads errors from database for paper_id
    - Gets current code from S3
    - Uses Bedrock to fix code
    - Tests fixed code with 2-minute timeout
    - If code runs 2 minutes without errors, consider it stable and stop reviewing
    - Otherwise, save error and continue fixing
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
        
        # Try to get errors from DynamoDB
        errors_list = get_errors(paper_id)
        error_count = len(errors_list)
        
        # Also check if error was passed directly in request (fallback if DynamoDB fails)
        direct_error = data.get('error_data')
        if direct_error and not errors_list:
            # DynamoDB might have failed, but we have error from previous iteration
            logger.info(f"DynamoDB unavailable, using error passed directly in request")
            errors_list = [{'error_data': direct_error}]
            error_count = 1
        
        logger.info(f"Retrieved {error_count} previous iterations for this paper.")
        
        # Safety limit to prevent infinite loops
        if error_count >= MAX_REVIEW_ITERATIONS:
            logger.warning(f"Max code review depth reached for {paper_id} ({error_count})")
            return jsonify({
                "success": False,
                "error": f"Max review iterations ({MAX_REVIEW_ITERATIONS}) reached",
                "paper_id": paper_id,
                "error_count": error_count
            }), 400
            
        # Build error message from all previous errors (not just latest)
        # This gives the AI context about what was tried before
        error_message = ""
        error_history = ""
        if errors_list:
            # Extract errors from all previous iterations
            all_error_messages = []
            for i, error_item in enumerate(errors_list):
                error_data = error_item.get('error_data', {})
                extracted = extract_errors_from_result(error_data)
                if extracted:
                    iteration_num = i + 1
                    all_error_messages.append(f"--- Iteration {iteration_num} Error ---\n{extracted}")
            
            if all_error_messages:
                # Use latest error as primary, but include history
                error_message = all_error_messages[-1]  # Latest error
                if len(all_error_messages) > 1:
                    # Include previous errors for context
                    error_history = "\n\n".join(all_error_messages[:-1])  # All except latest
                    logger.info(f"Including {len(all_error_messages) - 1} previous error(s) for context")
            else:
                logger.warning(f"No extractable errors found for {paper_id}, but errors list is not empty")
        
        # Get current code from S3
        code = get_code(paper_id)
        if not code:
            return jsonify({
                "success": False,
                "error": f"No code found in S3 for {paper_id}"
            }), 404
        
        # Get paper summary from OpenSearch for better context
        paper_summary = None
        if OPENSEARCH_AVAILABLE:
            try:
                opensearch_client = OpenSearchClient()
                paper = opensearch_client.get_paper_by_id(paper_id)
                if paper:
                    paper_summary = opensearch_client.get_paper_summary(paper)
                    logger.info(f"Retrieved paper summary for {paper_id} ({len(paper_summary)} chars)")
            except Exception as e:
                logger.warning(f"Could not retrieve paper summary for {paper_id}: {e}")
        
        # Get ALL errors from ALL papers for proactive error checking
        all_past_errors = get_all_errors(limit=200)  # Get up to 200 past errors from all papers
        logger.info(f"Retrieved {len(all_past_errors)} past errors from all papers for proactive checking")
        
        # If we have errors, fix the code with Bedrock
        if error_message:
            logger.info(f"Fixing code with Bedrock (iteration {error_count})...")
            fixed_code = fix_code_with_bedrock(code, error_message, error_count, paper_summary, error_history, all_past_errors)
            
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
            code = fixed_code  # Use fixed code for testing
        else:
            logger.info(f"No errors to fix, testing current code for stability...")
        
        # Test code with 2-minute timeout to check for immediate errors
        # If it runs for 2 minutes without errors, consider it stable
        logger.info(f"Testing code stability: running for {CODE_REVIEW_STABILITY_TIME}s to check for errors...")
        test_result = execute_code_sync(
            paper_id=f"{paper_id}_review_test",
            code=code,
            timeout=CODE_REVIEW_STABILITY_TIME,
            paper_title=data.get('paper_title')
        )
        
        # Check if code ran successfully for the stability period
        if test_result.get('success') and test_result.get('execution_time', 0) >= CODE_REVIEW_STABILITY_TIME - 10:
            # Code ran successfully for ~2 minutes - consider it stable
            logger.info(f"✅ Code is stable! Ran for {test_result.get('execution_time')}s without errors.")
            
            # Save stable code to S3 and trigger full execution
            s3_key = save_code(paper_id, code)
            logger.info(f"Stable code saved to S3: {s3_key}")
            
            # Trigger full execution with the stable code
            try:
                exec_result = execute_internal(
                    paper_id=paper_id,
                    code=code,
                    timeout=MAX_EXECUTION_TIME,
                    paper_title=data.get('paper_title')
                )
                
                return jsonify({
                    "success": True,
                    "message": f"Code is stable after {error_count} iterations. Full execution triggered.",
                    "paper_id": paper_id,
                    "iteration": error_count,
                    "stability_test_time": test_result.get('execution_time'),
                    "execution_status": exec_result
                })
            except Exception as e:
                logger.error(f"Failed to trigger full execution: {e}")
                return jsonify({
                    "success": False,
                    "error": f"Code is stable but failed to trigger execution: {str(e)}",
                    "paper_id": paper_id
                }), 500
        
        # Code failed or didn't run long enough - save error and continue in background thread
        logger.warning(f"Code test failed or didn't run long enough. Execution time: {test_result.get('execution_time', 0)}s")
        
        # Extract error from test result
        test_error_message = extract_errors_from_result(test_result)
        if not test_error_message:
            test_error_message = f"Code execution failed or timed out early (ran for {test_result.get('execution_time', 0)}s)"
        
        # Save error to DB
        error_data = {
            "stderr": test_result.get('stderr', ''),
            "stdout": test_result.get('stdout', ''),
            "error_message": test_error_message,
            "error_type": test_result.get('error_type', 'execution_error'),
            "return_code": test_result.get('return_code', -1),
            "execution_time": test_result.get('execution_time', 0)
        }
        save_error(paper_id, error_data)
        
        # Continue code review in background thread to avoid blocking and infinite loops
        def continue_review():
            try:
                logger.info(f"Continuing code review for {paper_id} (iteration {error_count + 1}) in background...")
                time.sleep(2)  # Small delay to avoid immediate retry
                
                # Pass error_data directly in case DynamoDB is unavailable
                review_payload = {
                    "paper_id": paper_id, 
                    "paper_title": data.get('paper_title'),
                    "error_data": error_data  # Pass error directly as fallback
                }
                
                review_response = requests.post(
                    f"http://localhost:8000/code_review",
                    json=review_payload,
                    timeout=300  # 5 minute timeout
                )
                logger.info(f"Code review iteration {error_count + 1} completed: {review_response.status_code}")
            except Exception as e:
                logger.error(f"Failed to continue code review in background: {e}")
        
        # Start background thread
        review_thread = threading.Thread(target=continue_review, daemon=True)
        review_thread.start()
        
        # Return immediately to avoid blocking
        return jsonify({
            "success": True,
            "message": f"Code review iteration {error_count + 1} triggered in background",
            "paper_id": paper_id,
            "iteration": error_count + 1,
            "test_result": test_result
        })
        
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


@app.route('/notify/<paper_id>', methods=['POST'])
def trigger_slack_notification(paper_id: str):
    """
    Manually trigger Slack notification for a paper.
    Useful if notification was missed during execution.
    """
    try:
        logger.info(f"Manually triggering Slack notification for {paper_id}")
        logger.info(f"OPENSEARCH_AVAILABLE: {OPENSEARCH_AVAILABLE}, SLACK_AVAILABLE: {SLACK_AVAILABLE}")
        logger.info(f"OpenSearchClient: {OpenSearchClient}, SlackNotifier: {SlackNotifier}")
        logger.info(f"OpenSearchClient is None: {OpenSearchClient is None}, SlackNotifier is None: {SlackNotifier is None}")
        
        if not SLACK_AVAILABLE or not OPENSEARCH_AVAILABLE or OpenSearchClient is None or SlackNotifier is None:
            return jsonify({
                "success": False,
                "error": f"Slack/OpenSearch not available (SLACK_AVAILABLE={SLACK_AVAILABLE}, OPENSEARCH_AVAILABLE={OPENSEARCH_AVAILABLE}, OpenSearchClient={OpenSearchClient}, SlackNotifier={SlackNotifier})",
                "paper_id": paper_id
            }), 503
        
        # Try to get execution result from S3 or reconstruct from status
        # For now, create a basic success result
        execution_result = {
            "success": True,
            "execution_time": 0,
            "return_code": 0,
            "stdout": "",
            "stderr": "",
            "error_message": ""
        }
        
        # Use the module-level classes
        send_slack_notification(paper_id, execution_result)
        
        return jsonify({
            "success": True,
            "message": f"Slack notification triggered for {paper_id}",
            "paper_id": paper_id
        })
    except Exception as e:
        logger.error(f"Error triggering Slack notification: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "paper_id": paper_id
        }), 500


if __name__ == '__main__':
    try:
        logger.info("Starting Trainium Executor Service v2")
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Working directory: {WORKING_DIR}")
        logger.info(f"Max execution time: {MAX_EXECUTION_TIME}s")
        logger.info(f"Success assumption time: {SUCCESS_ASSUMPTION_TIME}s")
        logger.info(f"Code storage bucket: papers-code-artifacts")
        logger.info(f"Max review iterations: {MAX_REVIEW_ITERATIONS}")
        
        # Validate critical directories exist and are writable
        try:
            os.makedirs(WORKING_DIR, exist_ok=True)
            test_file = os.path.join(WORKING_DIR, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"✅ Working directory is writable: {WORKING_DIR}")
        except Exception as e:
            logger.error(f"❌ Working directory not writable: {WORKING_DIR} - {e}")
            sys.exit(1)
        
        # Validate log directory
        try:
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"✅ Log directory ready: {log_dir}")
        except Exception as e:
            logger.error(f"❌ Cannot create log directory: {log_dir} - {e}")
            sys.exit(1)
        
        # Print all environment variables (but not sensitive ones)
        logger.info("=" * 80)
        logger.info("Environment Variables:")
        logger.info("=" * 80)
        for key, value in sorted(os.environ.items()):
            # Mask sensitive values
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'token', 'password', 'credential']):
                logger.info(f"{key}=***MASKED***")
            else:
                logger.info(f"{key}={value}")
        logger.info("=" * 80)
        
        # Validate AWS credentials are available (warn if not, but don't fail)
        try:
            import boto3
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"✅ AWS credentials available: {identity.get('Arn', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️  AWS credentials not available: {e}")
            logger.warning("   Some features (S3, DynamoDB, Bedrock) may not work")
        
        # Log OpenSearch/Slack availability
        logger.info("=" * 80)
        logger.info("Service Status:")
        logger.info(f"  OPENSEARCH_AVAILABLE: {OPENSEARCH_AVAILABLE}")
        logger.info(f"  SLACK_AVAILABLE: {SLACK_AVAILABLE}")
        logger.info(f"  Code gen directory: {code_gen_dir}")
        logger.info(f"  Code gen exists: {os.path.exists(code_gen_dir)}")
        if os.path.exists(code_gen_dir):
            logger.info(f"  Code gen files: {', '.join([f for f in os.listdir(code_gen_dir) if f.endswith('.py')][:5])}")
        logger.info("=" * 80)
        logger.info("Starting Flask server on 0.0.0.0:8000")
        logger.info("=" * 80)
        
        app.run(host='0.0.0.0', port=8000, threaded=True)
    except Exception as e:
        logger.error(f"FATAL: Failed to start application: {e}", exc_info=True)
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

