#!/usr/bin/env python3
import sys
import os

# Add script directory to Python path so we can import local modules
# This is critical when running via systemd where the working directory might differ
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

code_gen_dir = os.path.join(os.path.dirname(script_dir), 'code_gen')
home_code_gen = os.path.join(os.path.expanduser('~'), 'code_gen')

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
import signal
from datetime import datetime
from typing import Dict, List, Any, Optional
import psutil
import shutil
import boto3
from botocore.exceptions import ClientError
from error_db import save_error, get_errors, clear_errors, update_error_fixes, get_errors_for_paper_ids
from s3_code_storage import save_code, get_code, code_exists

app = Flask(__name__)

log_dir = os.path.join(os.path.expanduser('~'), 'trainium-executor', 'logs')
log_file = os.path.join(log_dir, 'trainium-executor.log')

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

# Attempt to Configure Slack, OpenSearch, and SageMaker metrics modules
try:
    from slack_notifier import SlackNotifier
    SLACK_AVAILABLE = True
except ImportError as e: 
    SLACK_AVAILABLE = False
    SlackNotifier = None
    logger.warning("Slack module not found - updates will be disabled.")
    
try:
    from opensearch_client import OpenSearchClient
    OPENSEARCH_AVAILABLE = True
except ImportError as e:
    OPENSEARCH_AVAILABLE = False
    OpenSearchClient = None
    logger.warning("OpenSearch client module not found - OpenSearch functionality will be disabled.")

try:
    from sagemaker_metrics import SageMakerMetricsLogger, create_metrics_logger
    SAGEMAKER_METRICS_ENABLED = True
except ImportError:
    SAGEMAKER_METRICS_ENABLED = False
    logger.warning("sagemaker_metrics module not found. Metrics logging to CloudWatch will be disabled.")

# Configuration
MAX_EXECUTION_TIME = int(os.getenv('MAX_EXECUTION_TIME', '1800'))  
SUCCESS_ASSUMPTION_TIME = int(os.getenv('SUCCESS_ASSUMPTION_TIME', '300'))  
WORKING_DIR = os.getenv('WORKING_DIR', '/tmp/trainium_jobs')
DATASET_CACHE_DIR = os.getenv('DATASET_CACHE_DIR', '/tmp/datasets')
NEURON_PROFILER_ENABLED = os.getenv('NEURON_PROFILER_ENABLED', 'true').lower() == 'true'
PROFILER_OUTPUT_DIR = os.getenv('PROFILER_OUTPUT_DIR', '/tmp/neuron_profiler')
RESULTS_BUCKET = os.getenv('RESULTS_BUCKET', 'trainium-execution-results')
MAX_REVIEW_ITERATIONS = int(os.getenv('MAX_REVIEW_ITERATIONS', '6'))  
ERROR_DB_TABLE_NAME = os.getenv('ERROR_DB_TABLE_NAME', 'docRunErrors') # Table structure: Partition key = DOC#<docId>, Sort key = ITER#<iteration>#ERR#<errorId>
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN', 'xoxb-552112250854-10031003801584-OFAzmiCTvAsECqlzIKmy9Ck1')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL', 'apl-research-papers')
TRAINIUM_EXECUTION_QUEUE_URL = os.getenv('TRAINIUM_EXECUTION_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo')
MAX_TRAINIUM_CONCURRENT = int(os.getenv('MAX_TRAINIUM_CONCURRENT', '1'))  # Max concurrent executions
QUEUE_POLL_INTERVAL = int(os.getenv('QUEUE_POLL_INTERVAL', '30'))  # Seconds between queue polls
ENABLE_AUTO_QUEUE_POLLING = os.getenv('ENABLE_AUTO_QUEUE_POLLING', 'true').lower() == 'true'  # Enable automatic queue polling

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

# Track running executions
running_executions = {}
execution_lock = threading.Lock()

# Track papers that have already had slack notifications sent
execution_notifications_sent = set()
final_code_notifications_sent = set()
notification_lock = threading.Lock()

# SQS client for queue polling
sqs_client = boto3.client('sqs', region_name=AWS_REGION) if TRAINIUM_EXECUTION_QUEUE_URL else None

def convert_to_perfetto_format(profiler_path: str) -> bool:
    if not os.path.exists(profiler_path):
        logger.warning(f"Profiler path does not exist: {profiler_path}")
        return False
    
    neuron_profile_cmd = '/opt/aws/neuron/bin/neuron-profile'
    if not os.path.exists(neuron_profile_cmd):
        logger.warning(f"neuron-profile command not found at {neuron_profile_cmd}, skipping Perfetto conversion")
        return False
    
    # Check if we have trace files to convert
    trace_files = []
    for root, dirs, files in os.walk(profiler_path):
        for file in files:
            if file.endswith('.pb') and file in ['ntrace.pb', 'cpu_util.pb', 'host_mem.pb']:
                trace_files.append(file)
    
    if not trace_files:
        logger.info(f"No trace files found in {profiler_path} to convert")
        return False
    
    logger.info(f"Found trace files to convert: {trace_files}")
    
    try:
        # Run neuron-profile view to convert trace files to Perfetto format
        # Using -d (directory) should automatically find and convert all trace files
        cmd = [
            neuron_profile_cmd,
            'view',
            '-d', profiler_path,
            '--output-format', 'perfetto'
        ]
        
        logger.info(f"Converting trace files to Perfetto format in {profiler_path}")
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for conversion
        )
        
        # Log output for debugging
        if result.stdout:
            logger.info(f"Conversion stdout: {result.stdout[:500]}")  # First 500 chars
        if result.stderr:
            logger.info(f"Conversion stderr: {result.stderr[:500]}")  # First 500 chars
        
        if result.returncode == 0:
            # Check if .pftrace files were actually created
            pftrace_files = []
            for root, dirs, files in os.walk(profiler_path):
                for file in files:
                    if file.endswith('.pftrace'):
                        pftrace_files.append(file)
            
            if pftrace_files:
                logger.info(f"✅ Successfully converted trace files to Perfetto format: {pftrace_files}")
                return True
            else:
                logger.warning(f"Conversion command succeeded but no .pftrace files found. Output: {result.stdout}")
                return False
        else:
            logger.warning(f"Failed to convert to Perfetto format (return code {result.returncode})")
            logger.warning(f"stderr: {result.stderr}")
            logger.warning(f"stdout: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout converting trace files to Perfetto format after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Error converting to Perfetto format: {e}")
        logger.error(traceback.format_exc())
        return False


def collect_profiler_artifacts(paper_id: str, profiler_path: str) -> Dict[str, Any]:
    if not os.path.exists(profiler_path):
        return {}
    
    profiler_info = {
        'profiler_enabled': True,
        'profiler_output_dir': profiler_path,
        'profiler_files': [],
        'profiler_s3_location': f"s3://{RESULTS_BUCKET}/profiler/{paper_id}/"
    }
    
    try:
        convert_to_perfetto_format(profiler_path)
        
        s3_client = boto3.client('s3')
        uploaded_files = []
        
        # Collect all files and prioritize .pftrace files
        pftrace_file = None
        for root, dirs, files in os.walk(profiler_path):
            for file in files:
                local_path = os.path.join(root, file)
                s3_key = f"profiler/{paper_id}/{file}"
                s3_client.upload_file(local_path, RESULTS_BUCKET, s3_key)
                uploaded_files.append(file)
                
                if file.endswith('.pftrace'):
                    pftrace_file = file
                elif not pftrace_file and (file.endswith('.perfetto') or 'perfetto' in file.lower()):
                    pftrace_file = file
                elif not pftrace_file and file == 'ntrace.pb':
                    pftrace_file = file
        
        if pftrace_file:
            profiler_info['perfetto_file'] = pftrace_file
            logger.info(f"Found Perfetto-compatible file: {pftrace_file}")
        
        profiler_info['profiler_files'] = uploaded_files
        logger.info(f"Uploaded {len(uploaded_files)} profiler artifacts for {paper_id}")
        
        # Update OpenSearch with profiler info
        if OPENSEARCH_AVAILABLE:
            try:
                opensearch_client = OpenSearchClient()
                opensearch_client.update_paper_execution_results(paper_id, {
                    "profiler_enabled": True,
                    "profiler_output_dir": profiler_path,
                    "profiler_files": uploaded_files,
                    "profiler_s3_location": profiler_info['profiler_s3_location'],
                    **({"profiler_perfetto_file": profiler_info['perfetto_file']} if profiler_info.get('perfetto_file') else {})
                })
                logger.info(f"✅ Updated OpenSearch with profiler info for {paper_id}")
            except Exception as e:
                logger.error(f"Error updating OpenSearch with profiler info: {e}")
        
        return profiler_info
    except Exception as e:
        logger.error(f"Failed to upload profiler artifacts: {e}")
        return {}

def upload_execution_results(paper_id: str, result: Dict[str, Any], executed_on_trn: bool = False):
    try:
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_key = f"results/{paper_id}/execution_result.json"
        s3_client.put_object(
            Bucket=RESULTS_BUCKET,
            Key=s3_key,
            Body=json.dumps(result, indent=2)
        )
        logger.info(f"Uploaded execution results to S3 for {paper_id}")
        
        # Update OpenSearch with execution results
        if OPENSEARCH_AVAILABLE:
            try:
                opensearch_client = OpenSearchClient()
                
                # Prepare execution results for OpenSearch
                execution_data = {
                    "execution_success": result.get('success', False),
                    "executed_on_trn": True,
                    "execution_time_seconds": round(result.get('execution_time', 0), 2),
                    "execution_return_code": result.get('return_code', -1),
                    "execution_error": result.get('error_message') if not result.get('success') else None,
                    "execution_error_type": result.get('error_type') if not result.get('success') else None,
                    "execution_completed_at": datetime.now().isoformat(),
                    "results_s3_location": f"s3://{RESULTS_BUCKET}/{s3_key}",
                }
                
                # Add metrics if available (check both 'metrics' and 'detailed_metrics')
                metrics = result.get('metrics') or result.get('detailed_metrics', {})
                if metrics:
                    execution_data["execution_metrics"] = metrics
                
                # Add profiler info if available
                profiler_info = result.get('profiler', {})
                if profiler_info and profiler_info.get('profiler_enabled'):
                    execution_data["profiler_enabled"] = True
                    execution_data["profiler_output_dir"] = profiler_info.get('profiler_output_dir')
                    execution_data["profiler_files"] = profiler_info.get('profiler_files', [])
                    if profiler_info.get('profiler_s3_location'):
                        execution_data["profiler_s3_location"] = profiler_info.get('profiler_s3_location')
                    if profiler_info.get('perfetto_file'):
                        execution_data["profiler_perfetto_file"] = profiler_info.get('perfetto_file')
                
                # Update OpenSearch
                success = opensearch_client.update_paper_execution_results(paper_id, execution_data)
                if success:
                    logger.info(f"✅ Updated OpenSearch with execution results for {paper_id}")
                else:
                    logger.warning(f"⚠️ Failed to update OpenSearch for {paper_id}")
            except Exception as e:
                logger.error(f"Error updating OpenSearch with execution results: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("OpenSearch not available - skipping execution results update")
            
    except Exception as e:
        logger.error(f"Failed to upload results: {e}")
        

def send_slack_notification(paper_id: str, execution_result: Dict[str, Any], thread_ts: Optional[str] = None):
    if not SLACK_AVAILABLE or not OPENSEARCH_AVAILABLE:
        logger.warning(f"Slack/OpenSearch not available - skipping notification for {paper_id}")
        logger.warning(f"SLACK_AVAILABLE: {SLACK_AVAILABLE}, OPENSEARCH_AVAILABLE: {OPENSEARCH_AVAILABLE}")
        return
    
    try:
        # Initialize clients
        opensearch_client = OpenSearchClient()
        slack_notifier = SlackNotifier(SLACK_BOT_TOKEN, SLACK_CHANNEL)
        
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
        
        # Check if this execution result was already sent (prevent duplicates after Flask restart)
        # Compare execution result timestamp with what's stored in OpenSearch
        current_execution_time = execution_result.get('execution_time', 0)
        stored_execution_time = paper.get('execution_time_seconds', 0)
        stored_execution_completed_at = paper.get('execution_completed_at')
        
        if stored_execution_completed_at and abs(current_execution_time - stored_execution_time) < 0.1:
            try:
                completed_time = datetime.fromisoformat(stored_execution_completed_at.replace('Z', '+00:00'))
                time_since_completion = (datetime.now(completed_time.tzinfo) - completed_time).total_seconds()
                # Only skip if completed more than 5 minutes ago
                if time_since_completion > 300:
                    logger.info(f"Skipping duplicate notification for {paper_id} - execution results already in OpenSearch (execution_time: {current_execution_time}s, completed {time_since_completion:.0f}s ago)")
                    return
                else:
                    logger.info(f"Execution results found in OpenSearch but completed recently ({time_since_completion:.0f}s ago) - sending notification anyway (may be retry)")
            except Exception as e:
                logger.warning(f"Could not parse completion time, sending notification anyway: {e}")
        
        if not thread_ts:
            thread_ts = paper.get('slack_thread_ts')
            if thread_ts:
                logger.info(f"Retrieved slack_thread_ts from OpenSearch for {paper_id}: {thread_ts}")
        
        fields_to_exclude = {
            'embedding', 'embeddings', 'vector', 'vectors', 
            'pdf_bytes', 'pdf_content', 'raw_content',
            's3_bucket', 's3_key' 
        }
        
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
        
        for key, value in paper.items():
            if key not in fields_to_exclude and key not in filtered_paper:
                if isinstance(value, str) and len(value) > 2000:
                    continue
                if isinstance(value, (dict, list)) and len(str(value)) > 2000:
                    continue
                filtered_paper[key] = value
        
        filtered_paper['execution_success'] = execution_result.get('success', False)
        filtered_paper['execution_time_seconds'] = round(execution_result.get('execution_time', 0), 2)
        filtered_paper['execution_return_code'] = execution_result.get('return_code', -1)
        
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
        
        
        code_s3_key = f"code/{paper_id}.py"
        filtered_paper['code_s3_location'] = f"s3://papers-code-artifacts/{code_s3_key}"
        filtered_paper['code_s3_url'] = f"https://s3.console.aws.amazon.com/s3/object/papers-code-artifacts?prefix={code_s3_key}"
        
        if execution_result.get('s3_results_key'):
            filtered_paper['results_s3_location'] = execution_result.get('s3_results_key')
        
        profiler_info = execution_result.get('profiler', {})
        if profiler_info and profiler_info.get('profiler_enabled'):
            filtered_paper['profiler_enabled'] = True
            filtered_paper['profiler_output_dir'] = profiler_info.get('profiler_output_dir')
            filtered_paper['profiler_files'] = profiler_info.get('profiler_files', [])
            if profiler_info.get('perfetto_file'):
                filtered_paper['profiler_perfetto_file'] = profiler_info.get('perfetto_file')
                profiler_s3_prefix = f"s3://{RESULTS_BUCKET}/profiler/{paper_id}/"
                filtered_paper['profiler_s3_location'] = profiler_s3_prefix
        
        # Send to Slack using custom formatting (reply in thread if thread_ts provided)
        success = slack_notifier.send_execution_notification(filtered_paper, execution_result, thread_ts=thread_ts)
        if success:
            logger.info(f"✅ Sent Slack notification for {paper_id}")
        else:
            logger.warning(f"Failed to send Slack notification for {paper_id}")
            
    except Exception as e:
        logger.error(f"Error sending Slack notification for {paper_id}: {e}")
        logger.error(traceback.format_exc())

    
def cleanup_stale_neuron_processes():
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            logger.warning("Could not list processes for cleanup")
            return
        
        lines = result.stdout.split('\n')
        neuron_processes = []
        
        for line in lines:
            if 'python' in line.lower() and ('neuron' in line.lower() or '/tmp/trainium_exec_' in line):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        # Skip our own process and the Flask app
                        if pid != os.getpid() and 'app.py' not in line:
                            neuron_processes.append(pid)
                    except (ValueError, IndexError):
                        continue
        
        if neuron_processes:
            logger.info(f"Found {len(neuron_processes)} stale Neuron processes, cleaning up...")
            for pid in neuron_processes:
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to stale process {pid}")
                except ProcessLookupError:
                    pass  # Process already gone
                except PermissionError:
                    logger.warning(f"Permission denied killing process {pid}")
        else:
            logger.debug("No stale Neuron processes found")
            
    except Exception as e:
        logger.warning(f"Error during Neuron process cleanup: {e}")


def execute_code_sync(paper_id: str, code: str, timeout: int, paper_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute code synchronously (blocking).
    This is the core execution function used by both /execute and /code_review.
    """
    # Clean up old processes before execution to free Neuron cores
    cleanup_stale_neuron_processes()
    
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
            'NEURON_RT_LOG_LEVEL': 'ERROR',
            'NEURON_RETRY_FAILED_COMPILATION': '1',  
            'NEURON_RT_NUM_CORES': '1', 
            'HUGGINGFACE_HUB_TOKEN': os.getenv('HUGGINGFACE_HUB_TOKEN', '')  
        }
        
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
        
        result = subprocess.run(
            cmd,
            cwd='/tmp',
            capture_output=True,
            text=True,
            timeout=timeout,
            env=exec_env,
            start_new_session=True  # Create new process group for proper cleanup
        )
        
        execution_time = time.time() - start_time
        
        # Remove from running executions
        with execution_lock:
            running_executions.pop(paper_id, None)
        
        # Check return code first
        success = result.returncode == 0
        
        # Check stderr for critical errors even if return code is 0
        # Some errors (like Neuron internal errors) may not set non-zero return code
        
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
        # Timeout occurred - subprocess.run() with start_new_session=True ensures
        # the process group (including child processes) is killed
        execution_time = time.time() - start_time
        logger.warning(f"Execution timeout ({timeout}s) reached for {paper_id} - process killed")
        
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
            "error_type": "timeout",
            "profiler_output_path": profiler_output_path
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
        
        # Restart Neuron runtime to free cores after each execution
        # This ensures cores are released even if the process didn't clean up properly
        try:
            logger.debug("Restarting Neuron runtime to free cores...")
            subprocess.run(
                ['sudo', 'systemctl', 'restart', 'neuron-rtd'],
                timeout=10,
                capture_output=True,
                check=False  # Don't fail if restart fails
            )
            logger.debug("Neuron runtime restarted successfully")
        except Exception as e:
            logger.warning(f"Failed to restart Neuron runtime (non-critical): {e}")


def extract_errors_from_result(exec_result: Dict[str, Any]) -> str:
    """Extract error message for display/logging (simplified)."""
    stderr = exec_result.get('stderr', '')
    message = exec_result.get('error_message', '')
    
    if not (stderr or message):
        return ''
    
    if exec_result.get('timeout') or exec_result.get('error_type') == 'timeout':
        return f"EXECUTION ERROR: Execution timed out - {exec_result.get('error_message', 'Timeout')}"
    
    error_message = (
        f"Error Message: {message if message else 'Code execution failed.'}. "
        f"Standard Error: {stderr if stderr else -1}"
    )
    
    return error_message

def format_full_error_context(error_data: Dict[str, Any], max_stderr_lines: int = 500, max_stdout_lines: int = 200) -> str:
    """
    Format full error context including complete traceback, stderr, and relevant stdout.
    Truncates intelligently to stay within context limits while preserving critical information.
    """
    stderr = error_data.get('stderr', '')
    stdout = error_data.get('stdout', '')
    error_message = error_data.get('error_message', '')
    return_code = error_data.get('return_code', -1)
    execution_time = error_data.get('execution_time', 0)
    
    parts = []
    
    # Header with basic info
    parts.append(f"Return Code: {return_code}")
    parts.append(f"Execution Time: {execution_time:.2f}s")
    if error_message:
        parts.append(f"Error Message: {error_message}")
    parts.append("")
    
    # Full stderr (with traceback) - this is most important
    if stderr:
        stderr_lines = stderr.split('\n')
        # Keep full traceback if present, truncate from top if too long
        if len(stderr_lines) > max_stderr_lines:
            # Keep last N lines (most recent errors) and first 50 lines (initial errors)
            first_part = '\n'.join(stderr_lines[:50])
            last_part = '\n'.join(stderr_lines[-max_stderr_lines+50:])
            parts.append("=== FULL STDERR (truncated, showing first 50 and last ~450 lines) ===")
            parts.append(first_part)
            parts.append(f"\n... ({len(stderr_lines) - max_stderr_lines} lines truncated) ...\n")
            parts.append(last_part)
        else:
            parts.append("=== FULL STDERR ===")
            parts.append(stderr)
        parts.append("")
    
    # Relevant stdout (last portion, often contains useful context)
    if stdout:
        stdout_lines = stdout.split('\n')
        if len(stdout_lines) > max_stdout_lines:
            # Keep last N lines (most recent output)
            relevant_stdout = '\n'.join(stdout_lines[-max_stdout_lines:])
            parts.append(f"=== STDOUT (last {max_stdout_lines} lines, {len(stdout_lines)} total) ===")
            parts.append(relevant_stdout)
        else:
            parts.append("=== FULL STDOUT ===")
            parts.append(stdout)
    
    return '\n'.join(parts)


def fix_code_with_bedrock(code: str, error_message: str, iteration: int, paper_id: str, errors_list: List[Dict[str, Any]], paper_summary: Optional[str] = None, similar_paper_errors: Optional[List[Dict[str, Any]]] = None, common_errors_all_papers: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:

    if not bedrock_client:
        logger.error("Bedrock client not available for code fixing")
        return None
    
    # Build error and fix history (excluding most recent error which doesn't have fixes yet)
    # Include more context but limit to recent iterations to stay within context window
    error_fix_history = ""
    if errors_list and len(errors_list) > 1:
        error_fix_history = "\n═══════════════════════════════════════════════════════════════════════════════\nPREVIOUS ERRORS AND FIXES (for context - learn from past mistakes):\n═══════════════════════════════════════════════════════════════════════════════\n"
        # Show last 3-4 previous errors (most relevant) with full context
        recent_errors = errors_list[:-1][-4:]  # Last 4 previous errors
        for i, error_item in enumerate(recent_errors, max(1, len(errors_list) - len(recent_errors))):
            error_data = error_item.get('error_data', {})
            iteration_num = error_item.get('iteration', i)
            
            error_fix_history += f"\n{'='*70}\nIteration {iteration_num} Error:\n{'='*70}\n"
            # Include full error context (stderr with traceback) but truncated intelligently
            full_error = format_full_error_context(error_data, max_stderr_lines=300, max_stdout_lines=100)
            error_fix_history += full_error
            
            # Show what fixes were applied
            fixes_applied = error_item.get('fixes_applied')
            if fixes_applied:
                if isinstance(fixes_applied, str):
                    try:
                        fixes_applied = json.loads(fixes_applied)
                    except json.JSONDecodeError:
                        fixes_applied = None
                if fixes_applied:
                    fixes_list = fixes_applied.get('fixes', []) if isinstance(fixes_applied, dict) else fixes_applied
                    if fixes_list:
                        fixes_str = '\n  - '.join(fixes_list[:5]) if isinstance(fixes_list, list) else str(fixes_list)
                        error_fix_history += f"\n\nFixes Applied:\n  - {fixes_str}\n"
            error_fix_history += "\n"
        
    # Build paper context section
    paper_context = ""
    if paper_summary:
        paper_context = f"""
PAPER CONTEXT (for better understanding of what the code should implement):
{paper_summary}

"""
    # Remove logger this once done debugging
    logger.info(f"Paper context: {paper_context}")

    # Build common errors context from ALL papers (most frequent patterns)
    common_errors_context = ""
    if common_errors_all_papers:
        common_errors_list = []
        for err_info in common_errors_all_papers[:15]:  # Top 15 most common
            error_text = err_info.get('error', '')
            frequency = err_info.get('frequency', 0)
            affected = err_info.get('affected_papers', 0)
            if error_text:
                common_errors_list.append(f"  - [{frequency}x in {affected} papers] {error_text}")
        
        if common_errors_list:
            common_errors_context = f"""
═══════════════════════════════════════════════════════════════════════════════
MOST COMMON ERRORS ACROSS ALL PAPERS (learned from DynamoDB):
═══════════════════════════════════════════════════════════════════════════════
These are the most frequent errors seen across all papers. PROACTIVELY check your code to prevent these:

{chr(10).join(common_errors_list)}

═══════════════════════════════════════════════════════════════════════════════
"""
        logger.info(f"Added {len(common_errors_list)} common error patterns to context")
    
    # Build similar paper errors context
    similar_errors_context = ""
    if similar_paper_errors:
        unique_errors = []
        seen_patterns = set()
        for similar_error in similar_paper_errors[:30]:
            error_data = similar_error.get('error_data', {})
            error_msg = error_data.get('error_message', '') or error_data.get('stderr', '')[:200]
            if error_msg:
                first_line = error_msg.split('\n')[0].strip()
                pattern_key = first_line[:100]
                if pattern_key and pattern_key not in seen_patterns:
                    seen_patterns.add(pattern_key)
                    similar_paper_id = similar_error.get('paper_id', 'unknown')
                    unique_errors.append(f"  - {first_line[:150]} (from paper: {similar_paper_id})")
        
        if unique_errors:
            similar_errors_context = f"""
═══════════════════════════════════════════════════════════════════════════════
ERRORS FROM SIMILAR PAPERS (proactive checking):
═══════════════════════════════════════════════════════════════════════════════
The following errors occurred in similar papers. Check your code to ensure these errors will NOT occur:

{chr(10).join(unique_errors[:30])}

═══════════════════════════════════════════════════════════════════════════════
"""
    if similar_paper_errors:
        logger.info(f"Similar errors context: {len(unique_errors)} unique patterns")
    
    # Brief essential guidance for cold start (when DynamoDB is empty) and ongoing review
    # DynamoDB error patterns will supplement this as more papers are tested
    review_guidance = """
═══════════════════════════════════════════════════════════════════════════════
ESSENTIAL TRAINIUM/XLA REQUIREMENTS (verify these in the code):
═══════════════════════════════════════════════════════════════════════════════
1. Device: 
   - MUST use: `import torch_xla.core.xla_model as xm` then `device = xm.xla_device()`
   - MUST NOT use: `torch_xla.device()` (THIS FUNCTION DOES NOT EXIST)
   - MUST NOT use: `torch.device('cuda')`, `torch.device('cpu')`, `.to('cuda')`, `.cuda()`
   - CRITICAL: If error shows "NameError: name 'torch_xla' is not defined", add the import:
     `import torch_xla.core.xla_model as xm` and use `xm.xla_device()`
2. Optimizer: `xm.optimizer_step(optimizer)` instead of `optimizer.step()`
3. Synchronization: 
   - DO NOT use `torch_xla.sync()` (THIS FUNCTION DOES NOT EXIST)
   - DO NOT use `xm.mark_step()` (deprecated)
   - Manual sync rarely needed - XLA syncs implicitly on .item() or logging
   - If explicit sync needed: use `xm.wait_device_ops()` (only at epoch boundaries or profiling)
4. Dataset: `train_loader, test_loader = load_dataset('name')` (returns EXACTLY 2 DataLoaders, NOT 3)
   - CRITICAL: If error shows "ValueError: not enough values to unpack (expected 3, got 2)", 
     fix unpacking to match: "train_loader, test_loader = load_dataset(...)" (NOT train_loader, val_loader, test_loader)
5. Neuron Core Allocation: 
   - CRITICAL: If error shows "Logical Neuron Core(s) not available - Requested:2 Available:0",
     remove ALL lines that set NEURON_RT_NUM_CORES (e.g., os.environ['NEURON_RT_NUM_CORES'] = '2')
   - Only 1 core is available - code must use default single-core allocation
   - If model is too large and requires 2+ cores, reduce model size/complexity (fewer layers, smaller hidden dims)
6. Imports: Use `from dataset_loader import load_dataset` (NOT torchvision.datasets)

Common mistakes to check:
- ❌ `torch_xla.device()` → ✅ `import torch_xla.core.xla_model as xm` then `xm.xla_device()` (torch_xla.device() DOES NOT EXIST)
- ❌ `device = torch_xla.device()` without import → ✅ Add `import torch_xla.core.xla_model as xm` first
- ❌ `torch_xla.sync()` → ✅ DO NOT use (function doesn't exist) - use `xm.wait_device_ops()` if needed
- ❌ `xm.mark_step()` → ✅ DO NOT use (deprecated) - XLA syncs implicitly
- ❌ `optimizer.step()` → ✅ `xm.optimizer_step(optimizer)` (CRITICAL: optimizer.step() does NOT work on Trainium - must use xm.optimizer_step)
- ❌ `xm.XlaModule` → ✅ `nn.Module`
- ❌ `xm.tensor()` → ✅ `torch.tensor()`
- ❌ `xm.dot_general()` → ✅ `torch.matmul()`
- ❌ `train_dataset, val_dataset, test_dataset = load_dataset()` → ✅ `train_loader, test_loader = load_dataset()`
- ❌ `train_loader, val_loader, test_loader = load_dataset()` → ✅ `train_loader, test_loader = load_dataset()` (only 2 values!)
- ❌ `item['key']` when item is a list → ✅ Check isinstance(item, dict) first
- ❌ `AutoTokenizer.from_pretrained('wikitext')` → ✅ Use dataset_loader (datasets come from S3, not HF Hub)
- ✅ `AutoModel.from_pretrained('bert-base-uncased')` → OK for fine-tuning (use public models)
- ❌ HTTP 401/403 errors → ✅ Use a public model instead (e.g., 'bert-base-uncased', 'gpt2')
- ❌ `base_model = ...` → ✅ Initialize with actual model
- ❌ Setting NEURON_RT_NUM_CORES → ✅ Use default single-core allocation
- ❌ CPU tensor indexing XLA tensor (RuntimeError: bridge::IsXlaTensor) → ✅ Move ALL tensors to device: data.to(device), labels.to(device), indices.to(device)
- ❌ `nn.Dropout2d` with 2D inputs → ✅ Use `nn.Dropout()` for 2D inputs (dropout2d requires 3D/4D)
- ❌ `nn.Dropout()` with 3D/4D inputs → ✅ Use `nn.Dropout2d()` for 3D/4D inputs (but this is rare - usually Dropout2d is wrong)
- ❌ `torchvision.transforms` → ✅ Use pure PyTorch operations (torch.tensor(), .view(), .reshape())
- ❌ Wrong input dimensions for model → ✅ Verify data shape matches model expectations, use .view()/.reshape() if needed
- ❌ Classifier dimension mismatch (INVALID_ARGUMENT: Cannot infer shape for dot operation) → ✅ Calculate flattened size correctly: after conv/pool layers, flattened = channels * height * width, then use `nn.Linear(flattened, num_classes)` NOT arbitrary numbers
- ❌ Error "f32[1,128] <dot> f32[16,10]" → ✅ Linear layer input_dim (16) doesn't match flattened features (128) - fix input_dim to match actual flattened size

CRITICAL: Verify fix direction is CORRECT:
- If code has `optimizer.step()` → MUST change to `xm.optimizer_step(optimizer)` (NOT the reverse!)
- If code has `nn.Dropout2d` with 2D inputs → MUST change to `nn.Dropout()` (NOT the reverse!)
- If code has `xm.xla_device()` → Keep it! Ensure `import torch_xla.core.xla_model as xm` exists (DO NOT change to torch_xla.device() - it doesn't exist!)
- If error shows "NameError: name 'torch_xla' is not defined" → Add `import torch_xla.core.xla_model as xm` and use `xm.xla_device()` (NOT torch_xla.device())

{error_patterns_note}
═══════════════════════════════════════════════════════════════════════════════
""".format(
        error_patterns_note="Note: Error patterns from DynamoDB shown above should be used to proactively prevent similar issues." 
        if (common_errors_context or similar_errors_context or error_fix_history) 
        else "Note: As more papers are tested, error patterns will be learned from DynamoDB and shown here."
    )
    
    prompt = f"""You are fixing PyTorch code that failed execution on AWS Trainium.

{paper_context}{common_errors_context}{similar_errors_context}{error_fix_history}═══════════════════════════════════════════════════════════════════════════════
CURRENT ERROR (most recent execution - FULL CONTEXT):
═══════════════════════════════════════════════════════════════════════════════
{error_message}

═══════════════════════════════════════════════════════════════════════════════
Current code (iteration {iteration}):
═══════════════════════════════════════════════════════════════════════════════
```python
{code}
```

{review_guidance}

═══════════════════════════════════════════════════════════════════════════════
COMPREHENSIVE ANALYSIS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════
Before fixing, perform a comprehensive analysis:

1. **Read the FULL error traceback** - Identify the exact line number and operation that failed
2. **Understand the root cause** - Don't just fix symptoms, address the underlying issue
3. **Check the error history above** - Learn from previous failed fixes to avoid repeating mistakes
4. **Review the entire code** - Look for related issues that might cause problems even if not currently failing
5. **Verify Trainium/XLA compatibility** - Ensure ALL operations are compatible with XLA (device placement, optimizer usage, etc.)
6. **Test your fix mentally** - Trace through the code to ensure your fix will actually work

IMPORTANT: Make ALL necessary fixes in one pass. Don't make partial fixes that will require another iteration.
═══════════════════════════════════════════════════════════════════════════════

CRITICAL: You must return BOTH the fixed code AND a summary of fixes in the following exact format:

```python
[FIXED CODE HERE - complete Python code]
```

---FIXES_SUMMARY_START---
[Summary of fixes made - concise list of what was changed]
---FIXES_SUMMARY_END---

The fixes summary should be a brief list (3-5 items max) of the key changes made. Example:
- Fixed XLA tensor size conversion by adding int() wrapper
- Replaced xm.optimizer.SGD with torch.optim.SGD
- Replaced xm.mark_step() with torch_xla.sync() (deprecated API)
- Fixed device placement: moved all tensors (data, labels, indices) to XLA device
- Fixed input dimensions: replaced nn.Dropout2d with nn.Dropout for 2D inputs

IMPORTANT: The code block must come FIRST, followed by the fixes summary between the markers.
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
        
        # Extract code and summary separately
        import re
        fixed_code = None
        fixes_summary = ["Code fixes applied via Bedrock"]  # Default
        
        # Extract code block (must come first)
        code_match = re.search(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
        if code_match:
            fixed_code = code_match.group(1).strip()
        else:
            # Fallback: try to extract code without markdown
            text_before_summary = generated_text.split('---FIXES_SUMMARY_START---')[0].strip()
            if text_before_summary.startswith('import') or text_before_summary.startswith('from'):
                fixed_code = text_before_summary
        
        # Extract fixes summary (between markers)
        summary_match = re.search(r'---FIXES_SUMMARY_START---\s*(.*?)\s*---FIXES_SUMMARY_END---', generated_text, re.DOTALL)
        if summary_match:
            summary_text = summary_match.group(1).strip()
            if summary_text:
                # Parse summary into list if it's bullet points or lines
                summary_lines = [line.strip().lstrip('- ').strip() for line in summary_text.split('\n') if line.strip()]
                if summary_lines:
                    fixes_summary = summary_lines
                else:
                    fixes_summary = [summary_text]
        
        if fixed_code:
            # Log fixes made
            code_changed = fixed_code != code
            if code_changed:
                logger.info(f"✅ Fixed code extracted (iteration {iteration}, length: {len(fixed_code)} chars)")
                logger.info(f"   Code length: {len(code)} → {len(fixed_code)} chars")
                summary_str = ', '.join(fixes_summary) if isinstance(fixes_summary, list) else str(fixes_summary)
                logger.info(f"   Fixes summary: {summary_str[:150]}")
            else:
                logger.warning(f"⚠️ Fixed code is identical to original (iteration {iteration})")
            
            return {
                "code": fixed_code,
                "fixes_summary": fixes_summary
            }
        
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


def execute_internal(paper_id: str, code: str, timeout: int, paper_title: Optional[str] = None, slack_thread_ts: Optional[str] = None, should_send_slack: bool = True, should_trigger_review: bool = True, iteration_num: Optional[int] = None) -> Dict[str, Any]:
    """
    Internal execute function (can be called directly or via HTTP endpoint).
    Returns dictionary instead of Flask response.
    
    Args:
        paper_id: Paper ID
        code: Code to execute
        timeout: Execution timeout
        paper_title: Optional paper title
        slack_thread_ts: Optional Slack thread timestamp for notifications
        should_send_slack: Whether to send Slack notification (True for final execution, False for code review tests)
        should_trigger_review: Whether to trigger code review on failure (False for final execution after max iterations)
        iteration_num: Current code review iteration number (used when triggering next iteration)
    """
    logger.info(f"Executing code for {paper_id} (should_send_slack={should_send_slack})")
    
    # Save code to S3 first
    s3_key = save_code(paper_id, code)
    if not s3_key:
        logger.warning(f"Failed to save code to S3 for {paper_id}, continuing anyway")
    
    # Execute code in background thread
    def execute_and_handle():
        exec_result = execute_code_sync(paper_id, code, timeout, paper_title)
        
        # Only send Slack notification for final execution results (not code review tests)
        # Check if we've already sent notification for this paper to prevent duplicates
        # Also check OpenSearch to prevent duplicates after Flask app restart
        if should_send_slack or exec_result.get('success'):
            with notification_lock:
                # Check in-memory set first (fast check)
                if paper_id in execution_notifications_sent:
                    logger.info(f"Execution notification already sent for {paper_id} (in-memory check) - skipping duplicate")
                else:
                    # Double-check OpenSearch to prevent duplicates after Flask restart
                    # The send_slack_notification function will also check, but we check here too for early exit
                    try:
                        opensearch_client = OpenSearchClient()
                        paper = opensearch_client.get_paper_by_id(paper_id)
                        if paper:
                            stored_execution_time = paper.get('execution_time_seconds', 0)
                            stored_execution_completed_at = paper.get('execution_completed_at')
                            current_execution_time = exec_result.get('execution_time', 0)
                            
                            # If OpenSearch has execution results with same execution time AND it was completed recently (within last hour),
                            # this notification was likely already sent. But be less aggressive - only skip if execution time matches exactly
                            # and it was completed more than 5 minutes ago (to allow retries for recent failures)
                            if stored_execution_completed_at and abs(current_execution_time - stored_execution_time) < 0.1:
                                try:
                                    completed_time = datetime.fromisoformat(stored_execution_completed_at.replace('Z', '+00:00'))
                                    time_since_completion = (datetime.now(completed_time.tzinfo) - completed_time).total_seconds()
                                    # Only skip if completed more than 5 minutes ago (allows retries for recent failures)
                                    if time_since_completion > 300:
                                        logger.info(f"Execution notification likely already sent for {paper_id} (OpenSearch check - execution_time: {current_execution_time}s, completed {time_since_completion:.0f}s ago) - skipping duplicate")
                                        execution_notifications_sent.add(paper_id)  # Add to set to prevent future checks
                                        return
                                    else:
                                        logger.info(f"Execution results found in OpenSearch but completed recently ({time_since_completion:.0f}s ago) - sending notification anyway (may be retry)")
                                except Exception as e:
                                    logger.warning(f"Could not parse completion time, sending notification anyway: {e}")
                    except Exception as e:
                        logger.warning(f"Could not check OpenSearch for duplicate notification: {e}")
                        # Continue anyway - send_slack_notification will also check
                    
                    logger.info(f"Execution completed for {paper_id}, sending final Slack notification...")
                    try:
                        send_slack_notification(paper_id, exec_result, thread_ts=slack_thread_ts)
                        execution_notifications_sent.add(paper_id)
                        logger.info(f"Final Slack notification sent for {paper_id}")
                    except Exception as e:
                        logger.error(f"Failed to send final Slack notification for {paper_id}: {e}")
                        logger.error(traceback.format_exc())
        else:
            logger.info(f"Execution completed for {paper_id} (code review test - skipping Slack notification)")
        
        if exec_result.get('success'):
            logger.info(f"Execution succeeded for {paper_id}")
            
            # Collect profiler artifacts first (if available) so we can include info in execution_result
            profiler_output_path = exec_result.get("profiler_output_path")
            if profiler_output_path:
                profiler_info = collect_profiler_artifacts(paper_id, profiler_output_path)
                if profiler_info:
                    # Add profiler info to execution_result before uploading
                    exec_result['profiler'] = profiler_info
            
            # Upload execution results (now includes profiler info if available)
            upload_execution_results(paper_id, exec_result)
            
            # Success - keep errors in DynamoDB for debugging and learning (never clear them)
            logger.info(f"Execution succeeded for {paper_id} - keeping all errors in DynamoDB for debugging")
        else:
            # Failure - handle based on whether we should trigger review
            if should_trigger_review:
                # Normal execution failure - save error and trigger code review
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
                # If iteration_num is provided, increment it for next iteration
                # Otherwise start with iteration 1 (first code review iteration)
                next_iteration = (iteration_num + 1) if iteration_num is not None else 1
                try:
                    review_payload = {
                        "paper_id": paper_id, 
                        "paper_title": paper_title,
                        "slack_thread_ts": slack_thread_ts,  # Pass thread_ts for Slack threading
                        "iteration_num": next_iteration,  # Pass incremented iteration number
                        "error_data": error_data  # Pass error directly in case DynamoDB fails
                    }
                    review_response = requests.post(
                        f"http://localhost:8000/code_review",
                        json=review_payload,
                        timeout=300  # 5 minute timeout for review
                    )
                    logger.info(f"Code review triggered for {paper_id} (iteration {next_iteration}): {review_response.status_code}")
                except Exception as e:
                    logger.error(f"Failed to trigger code review: {e}")
            else:
                # Final execution failure (after max iterations or after successful code review)
                # Don't trigger another review - just log the failure
                logger.warning(f"Final execution failed for {paper_id} (not triggering code review - already at max iterations or post-review)")
                # Still upload results even on failure (for debugging)
                upload_execution_results(paper_id, exec_result, executed_on_trn=True)
    
    # Start execution in background
    thread = threading.Thread(target=execute_and_handle, daemon=True)
    thread.start()
    
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
        "paper_title": "Paper Title",
        "slack_thread_ts": "1234567890.123456"  # Optional: Slack thread timestamp
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
        slack_thread_ts = data.get('slack_thread_ts')  # Optional Slack thread timestamp
        logger.info(f"Recieved code for {paper_id}, Title: {paper_title}")
        
        if not paper_id or not code:
            return jsonify({
                "success": False,
                "error": "paper_id and code are required"
            }), 400
        
        # Call internal execute function
        # NOTE: Initial execution from Lambda should NOT send Slack notifications
        # because this is not the final code - it will trigger code review if it fails.
        # Only the final execution (after code review) should send execution results.
        logger.info(f"Beginning internal execution (initial - will not send Slack notifications).")
        result = execute_internal(
            paper_id=paper_id,
            code=code,
            timeout=timeout,
            paper_title=paper_title,
            slack_thread_ts=slack_thread_ts,
            should_send_slack=False  # Initial execution is not final - don't send results yet
        )
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
        
        # Get iteration_num from request, default to 1 if not provided
        iteration_num = data.get('iteration_num', 1)
        if not isinstance(iteration_num, int) or iteration_num < 1:
            iteration_num = 1
        
        logger.info(f"Starting code review for {paper_id} (iteration {iteration_num})")
        
        # Try to get errors from DynamoDB
        errors_list = get_errors(paper_id)
        error_count = len(errors_list)
        
        logger.info(f"Retrieved {len(errors_list)} previous errors for this paper (iteration {iteration_num}).")
        
        # Safety limit to prevent infinite loops
        if iteration_num >= MAX_REVIEW_ITERATIONS:
            logger.warning(f"Max code review depth reached for {paper_id} (iteration {iteration_num}). Sending final code and execution results...")
            
            # Get final code from S3
            final_code = get_code(paper_id)
            if not final_code:
                logger.error(f"No code found in S3 for {paper_id} - cannot send final notifications")
                return jsonify({
                    "success": False,
                    "error": f"Max review iterations ({MAX_REVIEW_ITERATIONS}) reached, but no code found in S3.",
                    "paper_id": paper_id,
                    "iteration_num": iteration_num
                }), 400
            
            # Get code S3 key
            code_s3_key = f"code/{paper_id}.py"
            
            # Get paper title if available
            paper_title = data.get('paper_title')
            if not paper_title and OPENSEARCH_AVAILABLE:
                try:
                    opensearch_client = OpenSearchClient()
                    paper = opensearch_client.get_paper_by_id(paper_id)
                    if paper:
                        paper_title = paper.get('title')
                except Exception as e:
                    logger.warning(f"Could not get paper title from OpenSearch: {e}")
            
            # Send final code notification (second follow-up) - even though max iterations reached
            # IMPORTANT: Send this BEFORE triggering final execution to ensure correct order in Slack
            if SLACK_AVAILABLE:
                # Get slack_thread_ts from request or OpenSearch
                slack_thread_ts = data.get('slack_thread_ts')
                if not slack_thread_ts and OPENSEARCH_AVAILABLE:
                    try:
                        opensearch_client = OpenSearchClient()
                        paper = opensearch_client.get_paper_by_id(paper_id)
                        if paper:
                            slack_thread_ts = paper.get('slack_thread_ts')
                            if slack_thread_ts:
                                logger.info(f"Retrieved slack_thread_ts from OpenSearch for final code notification: {slack_thread_ts}")
                    except Exception as e:
                        logger.warning(f"Could not get slack_thread_ts from OpenSearch: {e}")
                
                if slack_thread_ts:
                    # Check if we've already sent a final code notification for this paper (prevent duplicates)
                    with notification_lock:
                        if paper_id in final_code_notifications_sent:
                            logger.info(f"Skipping duplicate final code notification for {paper_id} (max iterations) - already sent")
                        else:
                            try:
                                slack_notifier = SlackNotifier(SLACK_BOT_TOKEN, SLACK_CHANNEL)
                                final_code_sent = slack_notifier.send_final_code_notification(
                                    paper_id=paper_id,
                                    code_length=len(final_code),
                                    code_review_iterations=iteration_num,
                                    code_s3_key=code_s3_key,
                                    thread_ts=slack_thread_ts
                                )
                                if final_code_sent:
                                    logger.info(f"✅ Sent final code notification to Slack (max iterations reached)")
                                    final_code_notifications_sent.add(paper_id)
                                    # Small delay to ensure Slack processes the message before execution results
                                    time.sleep(1)
                                else:
                                    logger.warning(f"⚠️ Final code notification returned False")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to send final code notification: {e}")
                else:
                    logger.warning(f"⚠️ No slack_thread_ts available - skipping final code notification")
            
            # Trigger final execution with the final code (this will send execution results as third follow-up)
            # This happens AFTER the final code notification is sent to ensure correct order
            try:
                logger.info(f"Triggering final execution for {paper_id} (max iterations reached - using final code)")
                execute_internal(
                    paper_id=paper_id,
                    code=final_code,
                    timeout=MAX_EXECUTION_TIME,
                    paper_title=paper_title,
                    slack_thread_ts=data.get('slack_thread_ts'),
                    should_send_slack=True,  # This is the final execution - send notification
                    should_trigger_review=False  # Don't trigger review - already at max iterations
                )
                logger.info(f"Final execution triggered for {paper_id}")
            except Exception as e:
                logger.error(f"Failed to trigger final execution: {e}")
            
            # Keep errors in DynamoDB for debugging and learning (never clear them)
            logger.info(f"Keeping all errors in DynamoDB for {paper_id} (for debugging and learning)")
            
            return jsonify({
                "success": True,
                "message": f"Max review iterations ({MAX_REVIEW_ITERATIONS}) reached. Final code and execution results sent.",
                "paper_id": paper_id,
                "iteration_num": iteration_num,
                "final_code_sent": True,
                "final_execution_triggered": True
            }), 200
            
        # Extract full error context from most recent error (includes full stderr/stdout)
        error_message = ""
        full_error_context = ""
        if errors_list:
            latest_error = errors_list[-1]
            error_data = latest_error.get('error_data', {})
            # Get simplified message for logging
            error_message = extract_errors_from_result(error_data)
            # Get full context with traceback for the prompt
            full_error_context = format_full_error_context(error_data, max_stderr_lines=500, max_stdout_lines=200)
            if not error_message and not full_error_context:
                logger.warning(f"No extractable error found for {paper_id}")
        
        # Get current code from S3
        code = get_code(paper_id)
        if not code:
            return jsonify({
                "success": False,
                "error": f"No code found in S3 for {paper_id}"
            }), 404
        
        # Get paper summary and similar paper errors from OpenSearch
        paper_summary = None
        similar_paper_errors = []
        common_errors_all_papers = []
        
        # Get common errors from ALL papers in DynamoDB (proactive learning)
        try:
            from error_db import get_all_errors
            all_errors = get_all_errors(limit=100)  # Get recent errors from all papers
            if all_errors:
                # Extract unique error patterns
                error_patterns = {}
                for err in all_errors:
                    error_data = err.get('error_data', {})
                    error_msg = error_data.get('error_message', '') or error_data.get('stderr', '')
                    if error_msg:
                        # Extract first line as pattern key
                        first_line = error_msg.split('\n')[0].strip()
                        if first_line and len(first_line) > 20:  # Filter out very short errors
                            pattern_key = first_line[:150]  # Use first 150 chars as pattern
                            if pattern_key not in error_patterns:
                                error_patterns[pattern_key] = {
                                    'error': first_line[:200],
                                    'count': 0,
                                    'papers': set()
                                }
                            error_patterns[pattern_key]['count'] += 1
                            error_patterns[pattern_key]['papers'].add(err.get('paper_id', 'unknown'))
                
                # Sort by frequency and get top patterns
                sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1]['count'], reverse=True)
                common_errors_all_papers = [
                    {
                        'error': pattern_info['error'],
                        'frequency': pattern_info['count'],
                        'affected_papers': len(pattern_info['papers'])
                    }
                    for _, pattern_info in sorted_patterns[:20]  # Top 20 most common errors
                ]
                logger.info(f"Extracted {len(common_errors_all_papers)} common error patterns from all papers")
        except Exception as e:
            logger.warning(f"Could not extract common errors from all papers: {e}")
        
        if OPENSEARCH_AVAILABLE:
            try:
                opensearch_client = OpenSearchClient()
                similar_papers = opensearch_client.search_similar_papers(
                    paper_id=paper_id,
                    exclude_id=paper_id,
                    size=3
                )
                if similar_papers:
                    similar_paper_ids = [p.get('_id') for p in similar_papers if p.get('_id')]
                    logger.info(f"Found {len(similar_paper_ids)} similar papers: {', '.join(similar_paper_ids)}")
                    
                    # Get errors from similar papers
                    similar_paper_errors = get_errors_for_paper_ids(similar_paper_ids)
                    if similar_paper_errors:
                        logger.info(f"Retrieved {len(similar_paper_errors)} errors from similar papers")
            except Exception as e:
                logger.warning(f"Could not retrieve paper info: {e}. Check OpenSearch availability and credentials.")
        else:
            logger.warning(f"OpenSearch not available - skipping similar paper errors.")
        
        # If we have errors, fix the code with Bedrock
        if error_message or full_error_context:
            logger.info(f"Fixing code with Bedrock (iteration {iteration_num})...")
            # Pass full error context instead of just the message
            fix_result = fix_code_with_bedrock(code, full_error_context or error_message, iteration_num, paper_id, errors_list, paper_summary, similar_paper_errors, common_errors_all_papers)
            
            if not fix_result or not fix_result.get("code"):
                return jsonify({
                    "success": False,
                    "error": "Failed to fix code with Bedrock"
                }), 500
            
            fixed_code = fix_result["code"]
            fixes_summary = fix_result.get("fixes_summary", "Code fixes applied via Bedrock")
            
            # Save fixed code to S3 (replaces old)
            s3_key = save_code(paper_id, fixed_code)
            if not s3_key:
                return jsonify({
                    "success": False,
                    "error": "Failed to save fixed code to S3"
                }), 500
            
            logger.info(f"Fixed code saved to S3: {s3_key}")
            
            # Update most recent error with fixes_applied
            fixes_applied = {
                'iteration': iteration_num,  # Use iteration_num, not error_count
                'issues_found': [error_message[:200]],
                'fixes': fixes_summary if isinstance(fixes_summary, list) else [fixes_summary],
                'code_length_before': len(code),
                'code_length_after': len(fixed_code)
            }
            update_error_fixes(paper_id, fixes_applied)
            logger.info(f"Updated error record with fixes_applied for iteration {iteration_num}")
            
            code = fixed_code  # Use fixed code for testing
        else:
            logger.info(f"No errors to fix. Testing execution.")
        

        logger.info(f"Executing code with {MAX_EXECUTION_TIME}s for code review test (process will continue if no errors).")
        # Pass iteration_num so execute_internal can pass it to /code_review when execution fails
        test_result = execute_internal(
            paper_id=paper_id,
            code=code,
            timeout=MAX_EXECUTION_TIME,
            should_send_slack=False,  
            should_trigger_review=True,
            iteration_num=iteration_num  # Pass current iteration number
        )
        
        if test_result.get('success') and test_result.get('status') == 'running':
            return jsonify({
                "success": True,
                "message": f"Code review iteration {iteration_num + 1} triggered in background",
                "paper_id": paper_id,
                "iteration": iteration_num + 1,
                "test_result": test_result
            })
        else:
            return jsonify({
                "success": False,
                "error": "Code review test failed or didn't run long enough",
                "paper_id": paper_id,
                "iteration_num": iteration_num
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


def is_trainium_available() -> bool:
    """
    Check if Trainium is available (not currently executing papers).
    
    Returns:
        True if Trainium is available (no running executions), False otherwise
    """
    with execution_lock:
        current_executions = len(running_executions)
        is_available = current_executions < MAX_TRAINIUM_CONCURRENT
        logger.info(f"Trainium availability check: {current_executions} running, max {MAX_TRAINIUM_CONCURRENT}, available: {is_available}")
        return is_available


def poll_and_process_queue():
    """
    Poll the trainium-execution queue and process papers in batches.
    Only processes if Trainium is available.
    """
    if not sqs_client or not TRAINIUM_EXECUTION_QUEUE_URL:
        logger.warning("SQS client or queue URL not configured - queue polling disabled")
        return
    
    try:
        # Check if Trainium is available
        if not is_trainium_available():
            logger.info("Trainium is busy - skipping queue poll")
            return
        
        
        batch_size = MAX_TRAINIUM_CONCURRENT - len(running_executions)
        logger.info(f"Polling queue for up to {batch_size} papers...")
        response = sqs_client.receive_message(
            QueueUrl=TRAINIUM_EXECUTION_QUEUE_URL,
            MaxNumberOfMessages=batch_size,
            WaitTimeSeconds=0,  # Short polling (no long polling to avoid blocking)
            VisibilityTimeout=900  # 15 minutes (messages become visible again if not deleted)
        )
        
        messages = response.get('Messages', [])
        if not messages:
            logger.info("No messages in queue")
            return
        
        logger.info(f"Received {len(messages)} papers from queue")
        
        # Process each paper sequentially
        for message in messages:
            try:
                # Parse message body
                body = json.loads(message['Body'])
                paper_id = body.get('paper_id')
                code = body.get('code')
                paper_title = body.get('paper_title')
                slack_thread_ts = body.get('slack_thread_ts')
                
                if not paper_id or not code:
                    logger.warning(f"Invalid message: missing paper_id or code")
                    # Delete invalid message
                    sqs_client.delete_message(
                        QueueUrl=TRAINIUM_EXECUTION_QUEUE_URL,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                    continue
                
                # Check Trainium availability before processing
                if not is_trainium_available():
                    logger.info(f"Trainium became busy - stopping batch processing. Will retry {paper_id} later.")
                    # Don't delete message - let it become visible again for retry
                    break  # Stop processing remaining papers in this batch
                
                logger.info(f"Processing paper {paper_id} from queue")
                
                # Execute paper (this will handle code review internally if needed)
                # NOTE: Do NOT clear errors here - we need to preserve iteration history
                clear_errors(paper_id)

                result = execute_internal(
                    paper_id=paper_id,
                    code=code,
                    timeout=MAX_EXECUTION_TIME,
                    paper_title=paper_title,
                    slack_thread_ts=slack_thread_ts,
                    should_send_slack=False,  # Initial execution - code review will handle final notification
                    should_trigger_review=True
                )
                
                # Delete message from queue after successful processing
                # execute_internal returns status: "running" when execution starts successfully
                if result.get('success') and result.get('status') == 'running':
                    sqs_client.delete_message(
                        QueueUrl=TRAINIUM_EXECUTION_QUEUE_URL,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                    logger.info(f"✅ Paper {paper_id} accepted for execution, message deleted from queue")
                else:
                    logger.warning(f"Paper {paper_id} execution not accepted: {result}")
                    # Don't delete message - let it retry
                    
            except Exception as e:
                logger.error(f"Error processing message from queue: {e}")
                logger.error(traceback.format_exc())
                # Don't delete message on error - let it retry
        
    except ClientError as e:
        logger.error(f"AWS error polling queue: {e}")
    except Exception as e:
        logger.error(f"Error polling queue: {e}")
        logger.error(traceback.format_exc())


@app.route('/poll-queue', methods=['POST'])
def poll_queue_endpoint():
    """
    Endpoint to manually trigger queue polling.
    Useful for testing or external triggers.
    """
    try:
        poll_and_process_queue()
        return jsonify({
            "success": True,
            "message": "Queue polled successfully"
        })
    except Exception as e:
        logger.error(f"Error in poll-queue endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/queue-status', methods=['GET'])
def queue_status():
    """
    Get status of the execution queue.
    """
    if not sqs_client or not TRAINIUM_EXECUTION_QUEUE_URL:
        return jsonify({
            "success": False,
            "error": "Queue not configured"
        }), 503
    
    try:
        response = sqs_client.get_queue_attributes(
            QueueUrl=TRAINIUM_EXECUTION_QUEUE_URL,
            AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
        )
        attrs = response['Attributes']
        
        with execution_lock:
            current_executions = len(running_executions)
        
        return jsonify({
            "success": True,
            "queue_url": TRAINIUM_EXECUTION_QUEUE_URL,
            "messages_available": int(attrs.get('ApproximateNumberOfMessages', 0)),
            "messages_in_flight": int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0)),
            "current_executions": current_executions,
            "max_concurrent": MAX_TRAINIUM_CONCURRENT,
            "trainium_available": is_trainium_available()
        })
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
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
        
        # Start background thread for automatic queue polling
        if ENABLE_AUTO_QUEUE_POLLING and sqs_client and TRAINIUM_EXECUTION_QUEUE_URL:
            def queue_polling_worker():
                """Background worker that polls the queue periodically."""
                logger.info(f"Starting queue polling worker (interval: {QUEUE_POLL_INTERVAL}s)")
                while True:
                    try:
                        poll_and_process_queue()
                    except Exception as e:
                        logger.error(f"Error in queue polling worker: {e}")
                        logger.error(traceback.format_exc())
                    finally:
                        # Sleep before next poll
                        time.sleep(QUEUE_POLL_INTERVAL)
            
            polling_thread = threading.Thread(target=queue_polling_worker, daemon=True)
            polling_thread.start()
            logger.info("✅ Automatic queue polling enabled")
        else:
            if not ENABLE_AUTO_QUEUE_POLLING:
                logger.info("⚠️ Automatic queue polling disabled (ENABLE_AUTO_QUEUE_POLLING=false)")
            elif not sqs_client:
                logger.warning("⚠️ SQS client not available - automatic queue polling disabled")
            elif not TRAINIUM_EXECUTION_QUEUE_URL:
                logger.warning("⚠️ Queue URL not configured - automatic queue polling disabled")
        
        app.run(host='0.0.0.0', port=8000, threaded=True)
    except Exception as e:
        logger.error(f"FATAL: Failed to start application: {e}", exc_info=True)
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)