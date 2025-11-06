#!/usr/bin/env python3
"""
Trainium Executor Service

HTTP server that receives batches of PyTorch code from Lambda,
executes them on Trainium, and returns results.

This service runs on trn1.2xlarge instance and uses AWS Neuron SDK.
"""

from flask import Flask, request, jsonify
import subprocess
import tempfile
import os
import time
import logging
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import resource
import psutil
import shutil

app = Flask(__name__)

# Configure logging first
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

# Import SageMaker metrics logger (after logger is configured)
try:
    from sagemaker_metrics import SageMakerMetricsLogger, create_metrics_logger
    SAGEMAKER_METRICS_ENABLED = os.getenv('SAGEMAKER_METRICS_ENABLED', 'true').lower() == 'true'
except ImportError:
    SAGEMAKER_METRICS_ENABLED = False
    logger.warning("sagemaker_metrics module not found. Metrics logging to CloudWatch will be disabled.")

MAX_EXECUTION_TIME = int(os.getenv('MAX_EXECUTION_TIME', '600'))  # 10 minutes
WORKING_DIR = os.getenv('WORKING_DIR', '/tmp/trainium_jobs')
DATASET_CACHE_DIR = os.getenv('DATASET_CACHE_DIR', '/tmp/datasets')

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

def measure_resources_before():
    """Measure system resources before execution"""
    try:
        mem = psutil.virtual_memory()
        return {
            "memory_available_mb": mem.available / (1024 * 1024),
            "memory_used_mb": mem.used / (1024 * 1024)
        }
    except Exception as e:
        logger.warning(f"Failed to measure resources: {e}")
        return {}

def measure_resources_after():
    """Measure system resources after execution"""
    try:
        mem = psutil.virtual_memory()
        return {
            "memory_available_mb": mem.available / (1024 * 1024),
            "memory_used_mb": mem.used / (1024 * 1024),
            "peak_memory_mb": mem.used / (1024 * 1024)  # Simplified, ideally track peak
        }
    except Exception as e:
        logger.warning(f"Failed to measure resources: {e}")
        return {}

def setup_dataset_loader(job_dir: str):
    """Copy dataset_loader.py to job directory for use by generated code"""
    loader_source = os.path.join(os.path.dirname(__file__), 'dataset_loader.py')
    loader_dest = os.path.join(job_dir, 'dataset_loader.py')
    
    if os.path.exists(loader_source):
        shutil.copy2(loader_source, loader_dest)
        logger.debug(f"Copied dataset_loader.py to {job_dir}")
    else:
        logger.warning("dataset_loader.py not found, generated code won't have dataset utilities")

def execute_code(paper_id: str, code: str, timeout: int = MAX_EXECUTION_TIME, paper_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute Python code in isolated environment with Neuron support.
    
    Args:
        paper_id: Unique paper identifier
        code: Python code to execute
        timeout: Maximum execution time in seconds
        paper_title: Paper title (optional, for metrics logging)
        
    Returns:
        Dictionary with execution results
    """
    job_dir = os.path.join(WORKING_DIR, paper_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Copy dataset loader utilities to job directory
    setup_dataset_loader(job_dir)
    
    code_file = os.path.join(job_dir, 'main.py')
    
    # Initialize SageMaker metrics logger if enabled
    metrics_logger = None
    if SAGEMAKER_METRICS_ENABLED:
        try:
            instance_type = os.getenv('TRAINIUM_INSTANCE_TYPE', 'trn1.2xlarge')
            metrics_logger = create_metrics_logger(
                paper_id=paper_id,
                paper_title=paper_title,
                instance_type=instance_type
            )
            logger.info(f"Initialized SageMaker metrics logger for {paper_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize SageMaker metrics logger: {e}")
    
    try:
        with open(code_file, 'w') as f:
            f.write(code)
        
        logger.info(f"Executing code for paper {paper_id} (timeout: {timeout}s)")
        
        resources_before = measure_resources_before()
        start_time = time.time()

        result = subprocess.run(
            ['python3', 'main.py'],
            cwd=job_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                'NEURON_RT_LOG_LEVEL': 'ERROR',  
                'PYTHONUNBUFFERED': '1',
                'DATASET_CACHE_DIR': DATASET_CACHE_DIR
            }
        )
        
        execution_time = time.time() - start_time
        
        resources_after = measure_resources_after()
        
        success = result.returncode == 0
        
        metrics = extract_metrics_from_output(result.stdout)
        
        # Log metrics to SageMaker/CloudWatch if enabled
        if metrics_logger:
            try:
                # Extract and log training metrics from stdout
                metrics_logger.extract_and_log_metrics_from_output(result.stdout)
                
                # Log execution-level metrics
                additional_metrics = {k: v for k, v in metrics.items() 
                                    if k not in ['training_loss', 'accuracy', 'validation_accuracy', 
                                                'test_accuracy', 'epoch', 'step']}
                
                metrics_logger.log_execution_metrics(
                    execution_time=execution_time,
                    success=success,
                    peak_memory_mb=resources_after.get("peak_memory_mb"),
                    additional_metrics=additional_metrics
                )
                logger.info(f"Logged metrics to CloudWatch for {paper_id}")
            except Exception as e:
                logger.warning(f"Failed to log metrics to CloudWatch: {e}")
        
        execution_result = {
            "success": success,
            "execution_time": round(execution_time, 2),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timeout": False,
            "peak_memory_mb": resources_after.get("peak_memory_mb"),
            **metrics
        }
        
        if success:
            logger.info(f"Paper {paper_id} executed successfully in {execution_time:.1f}s")
        else:
            logger.warning(f"Paper {paper_id} failed with return code {result.returncode}")
        
        return execution_result
        
    except subprocess.TimeoutExpired as e:
        execution_time = time.time() - start_time
        logger.error(f"⏱Paper {paper_id} timed out after {timeout}s")
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
        logger.error(f"Error executing code for paper {paper_id}: {e}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "execution_time": 0,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "timeout": False,
            "error_message": f"Execution error: {str(e)}",
            "error_type": "execution_error"
        }
    
    finally:
        # Cleanup
        try:
            if os.path.exists(code_file):
                os.remove(code_file)
        except Exception as e:
            logger.warning(f"Failed to cleanup {code_file}: {e}")

def extract_metrics_from_output(stdout: str) -> Dict[str, Any]:
    """
    Extract metrics from code output.
    
    Looks for JSON lines starting with "METRICS:" in stdout.
    Example: print(f"METRICS: {json.dumps({'training_loss': 0.023, 'accuracy': 0.95})}")
    
    Args:
        stdout: Standard output from code execution
        
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {}
    
    try:
        for line in stdout.split('\n'):
            if line.startswith('METRICS:'):
                json_str = line.replace('METRICS:', '').strip()
                parsed_metrics = json.loads(json_str)
                metrics.update(parsed_metrics)
    except Exception as e:
        logger.warning(f"Failed to extract metrics from output: {e}")
    
    return metrics

def count_lines_of_code(code: str) -> int:
    """Count non-empty, non-comment lines of code"""
    lines = code.split('\n')
    loc = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            loc += 1
    return loc

def analyze_code(code: str) -> Dict[str, Any]:
    """Analyze code to extract metadata"""
    code_lower = code.lower()
    
    return {
        "lines_of_code": count_lines_of_code(code),
        "has_training_loop": any(x in code_lower for x in ['for epoch', 'train(', 'training_loop']),
        "has_evaluation": any(x in code_lower for x in ['eval(', 'evaluate(', 'test(']),
        "uses_distributed_training": any(x in code_lower for x in ['distributeddataparallel', 'ddp', 'torch.distributed']),
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "neuron_available": os.path.exists('/opt/aws/neuron'),
        "working_dir": WORKING_DIR,
        "max_execution_time": MAX_EXECUTION_TIME
    })

@app.route('/execute_batch', methods=['POST'])
def execute_batch():
    """
    Execute a batch of code files.
    
    Request body:
    {
        "batch": [
            {
                "paper_id": "paper_123",
                "paper_title": "ResNet Paper",
                "code": "import torch\n...",
                "s3_code_key": "paper_123/code.py"
            },
            ...
        ],
        "timeout": 600  # Optional, per-code timeout
    }
    
    Response:
    {
        "success": true,
        "batch_size": 10,
        "results": {
            "paper_123": {
                "success": true,
                "execution_time": 45.2,
                "return_code": 0,
                "stdout": "...",
                "stderr": "",
                ...
            },
            ...
        }
    }
    """
    try:
        data = request.get_json()
        batch = data.get('batch', [])
        timeout = data.get('timeout', MAX_EXECUTION_TIME)
        
        if not batch:
            return jsonify({
                "success": False,
                "error": "Empty batch"
            }), 400
        
        logger.info(f"📦 Received batch of {len(batch)} code files")
        
        results = {}
        
        for item in batch:
            paper_id = item['paper_id']
            paper_title = item.get('paper_title', 'Unknown')
            code = item['code']
            
            logger.info(f"Executing: {paper_title} ({paper_id})")
 
            code_analysis = analyze_code(code)
   
            exec_result = execute_code(paper_id, code, timeout, paper_title=paper_title)

            exec_result.update(code_analysis)
            
            results[paper_id] = exec_result

        successful = sum(1 for r in results.values() if r['success'])
        failed = len(results) - successful
        
        logger.info(f"Batch complete: {successful} succeeded, {failed} failed")
        
        return jsonify({
            "success": True,
            "batch_size": len(batch),
            "successful": successful,
            "failed": failed,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/execute', methods=['POST'])
def execute_single():
    """
    Execute a single code file (for testing).
    
    Request body:
    {
        "paper_id": "paper_123",
        "code": "import torch\n..."
    }
    """
    try:
        data = request.get_json()
        paper_id = data.get('paper_id')
        code = data.get('code')
        timeout = data.get('timeout', MAX_EXECUTION_TIME)
        
        if not paper_id or not code:
            return jsonify({
                "success": False,
                "error": "paper_id and code are required"
            }), 400
        
        result = execute_code(paper_id, code, timeout)
        result.update(analyze_code(code))
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error executing single code: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Trainium Executor Service")
    logger.info(f"Working directory: {WORKING_DIR}")
    logger.info(f"Max execution time: {MAX_EXECUTION_TIME}s")
    
    app.run(host='0.0.0.0', port=8000, threaded=True)

