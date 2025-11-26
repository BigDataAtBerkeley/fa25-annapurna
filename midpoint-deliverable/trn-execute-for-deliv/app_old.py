#!/usr/bin/env python3
"""
Trainium Executor Service

HTTP server that receives batches of PyTorch code from Lambda,
executes them on Trainium, and returns results.

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

MAX_EXECUTION_TIME = int(os.getenv('MAX_EXECUTION_TIME', '1800'))  # 30 minutes (increased for Neuron compilation)
WORKING_DIR = os.getenv('WORKING_DIR', '/tmp/trainium_jobs')
DATASET_CACHE_DIR = os.getenv('DATASET_CACHE_DIR', '/tmp/datasets')
NEURON_PROFILER_ENABLED = os.getenv('NEURON_PROFILER_ENABLED', 'true').lower() == 'true'
PROFILER_OUTPUT_DIR = os.getenv('PROFILER_OUTPUT_DIR', '/tmp/neuron_profiler')

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
os.makedirs(PROFILER_OUTPUT_DIR, exist_ok=True)

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

def ensure_synthetic_dataset():
    """
    Ensure synthetic dataset is downloaded from S3.
    This is called before execution to prevent missing dataset errors.
    """
    try:
        # Import dataset_loader from the same directory as this script
        loader_path = os.path.join(os.path.dirname(__file__), 'dataset_loader.py')
        if not os.path.exists(loader_path):
            logger.warning("dataset_loader.py not found, cannot download synthetic dataset")
            return False
        
        # Add the directory to sys.path temporarily for import
        import sys
        script_dir = os.path.dirname(__file__)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        try:
            from dataset_loader import DatasetManager
            
            manager = DatasetManager(cache_dir=DATASET_CACHE_DIR)
            
            # Check if synthetic dataset exists
            synthetic_dir = os.path.join(DATASET_CACHE_DIR, 'synthetic')
            required_files = ['synthetic_small.pt', 'synthetic_medium.pt', 'synthetic_tabular.pt']
            all_exist = all(os.path.exists(os.path.join(synthetic_dir, f)) for f in required_files)
            
            if all_exist:
                logger.info("✓ Synthetic dataset already available")
                return True
            
            # Download synthetic dataset
            logger.info("Synthetic dataset not found, downloading from S3...")
            try:
                dataset_dir = manager.download_dataset('synthetic', force=False)
                logger.info(f"✓ Synthetic dataset downloaded to {dataset_dir}")
                
                # Verify all files were downloaded
                all_exist = all(os.path.exists(os.path.join(dataset_dir, f)) for f in required_files)
                if all_exist:
                    logger.info(f"✓ Verified all synthetic dataset files are present")
                    return True
                else:
                    missing = [f for f in required_files if not os.path.exists(os.path.join(dataset_dir, f))]
                    logger.error(f"✗ Some synthetic dataset files are missing: {missing}")
                    return False
            except Exception as e:
                logger.error(f"Failed to download synthetic dataset: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("  Code requiring synthetic dataset will fail")
                return False
        finally:
            # Remove the directory from sys.path if we added it
            if script_dir in sys.path:
                sys.path.remove(script_dir)
            
    except Exception as e:
        logger.error(f"Failed to ensure synthetic dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def execute_code(paper_id: str, code: str, timeout: int = MAX_EXECUTION_TIME, paper_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute Python code in isolated environment with Neuron support.
    
    NEW APPROACH: Use a completely fresh temporary directory that only contains
    main.py and dataset_loader.py. Run Python from that directory so it finds
    dataset_loader.py automatically without needing PYTHONPATH manipulation.
    
    Args:
        paper_id: Unique paper identifier
        code: Python code to execute
        timeout: Maximum execution time in seconds
        paper_title: Paper title (optional, for metrics logging)
        
    Returns:
        Dictionary with execution results
    """
    # Create a completely fresh temporary directory for this execution
    # This ensures no torch conflicts or leftover files
    exec_dir = tempfile.mkdtemp(prefix=f'trainium_exec_{paper_id}_', dir='/tmp')
    
    try:
        # Write main.py to the fresh execution directory
        # Add sys.path manipulation to ensure correct imports
        main_py_path = os.path.join(exec_dir, 'main.py')
        with open(main_py_path, 'w') as f:
            f.write(f"""
import sys
import os
# Remove exec_dir from sys.path if it's there (shouldn't be, but just in case)
if '{exec_dir}' in sys.path:
    sys.path.remove('{exec_dir}')
# Add exec_dir to the END of sys.path for dataset_loader imports
# This ensures venv site-packages (where torch is) are searched first
sys.path.append('{exec_dir}')

# Now execute the actual code
{code}
""")
        
        # Copy dataset_loader.py to the execution directory
        # This way it's in the same directory as main.py and can be imported normally
        loader_source = os.path.join(os.path.dirname(__file__), 'dataset_loader.py')
        loader_dest = os.path.join(exec_dir, 'dataset_loader.py')
        if os.path.exists(loader_source):
            shutil.copy2(loader_source, loader_dest)
        else:
            logger.warning("dataset_loader.py not found, generated code won't have dataset utilities")
        
        # CRITICAL: Check for and remove any torch-related directories/files in exec_dir
        # Python adds exec_dir to sys.path[0] when running a script, which can cause
        # PyTorch to find a local torch directory instead of the installed package
        for item in os.listdir(exec_dir):
            item_path = os.path.join(exec_dir, item)
            if item.lower() in ['torch', '_torch', 'torch.py', '_torch.py']:
                logger.warning(f"Found conflicting {item} in exec_dir, removing it")
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
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
        
        # Ensure synthetic dataset is available only if code uses it
        # Check if code references 'synthetic' dataset to avoid unnecessary downloads
        code_lower = code.lower()
        if 'synthetic' in code_lower and ('load_dataset' in code_lower or 'dataset' in code_lower):
            ensure_synthetic_dataset()
        
        logger.info(f"Executing code for paper {paper_id} in isolated directory {exec_dir} (timeout: {timeout}s)")
        
        resources_before = measure_resources_before()
        start_time = time.time()

        # Set up environment with Neuron runtime in PATH
        user_home = os.path.expanduser('~')
        neuron_bin_paths = [
            '/opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin',
            '/opt/aws_neuronx_venv_pytorch_2_8/bin',
            '/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin',
            f'{user_home}/.local/bin',
            '/opt/aws/neuron/bin',
            '/usr/local/bin',
            '/usr/bin',
            '/bin'
        ]
        current_path = os.environ.get('PATH', '')
        neuron_path = ':'.join(neuron_bin_paths + [current_path])
        python_path = os.environ.get('PYTHONPATH', '')
        neuron_python = '/opt/aws_neuronx_venv_pytorch_2_8_nxd_training/bin/python3'
        
        # Set up Neuron Profiler if enabled
        profiler_output_path = None
        if NEURON_PROFILER_ENABLED:
            profiler_output_path = os.path.join(PROFILER_OUTPUT_DIR, f'{paper_id}_{int(time.time())}')
            os.makedirs(profiler_output_path, exist_ok=True)
            logger.info(f"Neuron Profiler enabled - output will be saved to {profiler_output_path}")
        
        # Build environment for execution
        exec_env = {
            **os.environ,
            'PATH': neuron_path,
            'PYTHONPATH': pythonpath,  # Keep minimal - venv Python already has correct paths
            'NEURON_RT_LOG_LEVEL': 'ERROR',  
            'PYTHONUNBUFFERED': '1',
            'DATASET_CACHE_DIR': DATASET_CACHE_DIR,
            'PYTHONDONTWRITEBYTECODE': '1'
        }
        
        # Use neuron-profile inspect to wrap execution if profiler is enabled
        # This provides hardware-level profiling for Trainium devices
        # neuron-profile inspect runs the command and captures hardware profiling data
        if NEURON_PROFILER_ENABLED:
            neuron_profile_cmd = '/opt/aws/neuron/bin/neuron-profile'
            if os.path.exists(neuron_profile_cmd):
                # Use neuron-profile inspect to wrap Python execution
                # This captures hardware-level profiling data (Neuron cores, memory, etc.)
                cmd = [
                    neuron_profile_cmd,
                    'inspect',
                    '-o', profiler_output_path,
                    neuron_python, main_py_path
                ]
                logger.info(f"Neuron Profiler enabled - using neuron-profile inspect to capture hardware profiling to {profiler_output_path}")
            else:
                # Fallback if neuron-profile not found
                logger.warning(f"neuron-profile not found at {neuron_profile_cmd}, running without profiler")
                cmd = [neuron_python, main_py_path]
        else:
            cmd = [neuron_python, main_py_path]
        
        result = subprocess.run(
            cmd,
            cwd='/tmp',  # Run from /tmp
            capture_output=True,
            text=True,
            timeout=timeout,
            env=exec_env
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
        
        # Collect profiler results if available
        # neuron-profile inspect creates subdirectories (instance_id_pid), so we need to search recursively
        profiler_results = None
        if NEURON_PROFILER_ENABLED and profiler_output_path and os.path.exists(profiler_output_path):
            profiler_files = []
            profiler_subdirs = []
            try:
                # Search recursively for profiler files
                for root, dirs, files in os.walk(profiler_output_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.isfile(file_path):
                            # Store relative path from profiler_output_path
                            rel_path = os.path.relpath(file_path, profiler_output_path)
                            profiler_files.append(rel_path)
                    # Track subdirectories (neuron-profile creates instance_id_pid subdirs)
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if os.path.isdir(dir_path):
                            rel_dir = os.path.relpath(dir_path, profiler_output_path)
                            profiler_subdirs.append(rel_dir)
                
                if profiler_files:
                    # Convert profiler files to Perfetto format
                    perfetto_file_path = None
                    try:
                        # Find the subdirectory with profiler files (usually instance_id_pid)
                        if profiler_subdirs:
                            profiler_subdir = profiler_subdirs[0]  # Use first subdirectory
                            profiler_session_dir = os.path.join(profiler_output_path, profiler_subdir)
                            
                            if os.path.exists(profiler_session_dir):
                                # Convert to Perfetto format
                                perfetto_file = os.path.join(profiler_output_path, f"{paper_id}_profile.pftrace")
                                neuron_profile_cmd = '/opt/aws/neuron/bin/neuron-profile'
                                
                                if os.path.exists(neuron_profile_cmd):
                                    convert_cmd = [
                                        neuron_profile_cmd,
                                        'view',
                                        '--session-dir', profiler_session_dir,
                                        '--output-format', 'perfetto',
                                        '--output-file', perfetto_file
                                    ]
                                    
                                    convert_result = subprocess.run(
                                        convert_cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=300  # 5 minute timeout for conversion
                                    )
                                    
                                    if convert_result.returncode == 0 and os.path.exists(perfetto_file):
                                        perfetto_file_path = perfetto_file
                                        file_size = os.path.getsize(perfetto_file) / (1024 * 1024)
                                        logger.info(f"✓ Converted profiler to Perfetto format: {perfetto_file} ({file_size:.2f} MB)")
                                    else:
                                        logger.warning(f"Failed to convert profiler to Perfetto: {convert_result.stderr}")
                                else:
                                    logger.warning("neuron-profile command not found, cannot convert to Perfetto")
                    except Exception as e:
                        logger.warning(f"Failed to convert profiler to Perfetto format: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                    
                    profiler_results = {
                        "profiler_output_dir": profiler_output_path,
                        "profiler_files": profiler_files,
                        "profiler_subdirs": profiler_subdirs,
                        "profiler_enabled": True,
                        "total_files": len(profiler_files),
                        "perfetto_file": perfetto_file_path  # Path to Perfetto file on Trainium
                    }
                    logger.info(f"Neuron Profiler results available: {len(profiler_files)} files in {profiler_output_path}")
                else:
                    logger.warning(f"Neuron Profiler directory exists but contains no files: {profiler_output_path}")
            except Exception as e:
                logger.warning(f"Failed to collect profiler results: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        execution_result = {
            "success": success,
            "execution_time": round(execution_time, 2),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timeout": False,
            "peak_memory_mb": resources_after.get("peak_memory_mb"),
            "detailed_metrics": metrics,  # Store metrics in detailed_metrics for test script
            "profiler": profiler_results,  # Neuron Profiler results
            **metrics  # Also spread for backward compatibility
        }
        
        if success:
            logger.info(f"Paper {paper_id} executed successfully in {execution_time:.1f}s")
        else:
            logger.warning(f"Paper {paper_id} failed with return code {result.returncode}")
            # Log detailed error information
            if result.stderr:
                logger.error(f"STDERR for {paper_id}:\n{result.stderr}")
            if result.stdout:
                # Log last 50 lines of stdout for context
                stdout_lines = result.stdout.split('\n')
                last_lines = '\n'.join(stdout_lines[-50:])
                logger.error(f"Last 50 lines of STDOUT for {paper_id}:\n{last_lines}")
        
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
        # Cleanup: Remove the entire temporary execution directory
        try:
            if os.path.exists(exec_dir):
                shutil.rmtree(exec_dir)
                logger.debug(f"Cleaned up execution directory: {exec_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup execution directory {exec_dir}: {e}")

def extract_metrics_from_output(stdout: str) -> Dict[str, Any]:
    """
    Extract metrics from code output.
    
    Looks for JSON lines starting with "METRICS:" in stdout.
    Also parses common printed patterns like "Epoch X/Y, Average Loss: X.XXXX" and "Test Accuracy: XX.XX%"
    
    Args:
        stdout: Standard output from code execution
        
    Returns:
        Dictionary of extracted metrics
    """
    metrics = {}
    import re
    
    try:
        # First, try to find METRICS: lines (preferred format)
        for line in stdout.split('\n'):
            if line.startswith('METRICS:'):
                json_str = line.replace('METRICS:', '').strip()
                parsed_metrics = json.loads(json_str)
                metrics.update(parsed_metrics)
        
        # If no METRICS: lines found, parse common printed patterns
        if not metrics:
            # Extract epoch losses: "Epoch X/Y, Average Loss: X.XXXX"
            epoch_losses = []
            for line in stdout.split('\n'):
                match = re.search(r'Epoch\s+\d+/\d+.*?Loss:\s*([\d.]+)', line, re.IGNORECASE)
                if match:
                    epoch_losses.append(float(match.group(1)))
            
            if epoch_losses:
                metrics['training_loss'] = epoch_losses[-1]  # Last epoch loss
                metrics['final_epoch_loss'] = epoch_losses[-1]
                metrics['initial_epoch_loss'] = epoch_losses[0] if epoch_losses else None
                metrics['num_epochs'] = len(epoch_losses)
            
            # Extract test accuracy: "Test Accuracy: XX.XX%"
            for line in stdout.split('\n'):
                match = re.search(r'Test\s+Accuracy:\s*([\d.]+)%', line, re.IGNORECASE)
                if match:
                    metrics['test_accuracy'] = float(match.group(1))
                    break
            
            # Extract training accuracy if present
            for line in stdout.split('\n'):
                match = re.search(r'Training\s+Accuracy:\s*([\d.]+)%', line, re.IGNORECASE)
                if match:
                    metrics['training_accuracy'] = float(match.group(1))
                    break
            
            # Extract validation accuracy if present
            for line in stdout.split('\n'):
                match = re.search(r'Validation\s+Accuracy:\s*([\d.]+)%', line, re.IGNORECASE)
                if match:
                    metrics['validation_accuracy'] = float(match.group(1))
                    break
            
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
    # Check synthetic dataset availability
    synthetic_dir = os.path.join(DATASET_CACHE_DIR, 'synthetic')
    required_files = ['synthetic_small.pt', 'synthetic_medium.pt', 'synthetic_tabular.pt']
    synthetic_available = all(os.path.exists(os.path.join(synthetic_dir, f)) for f in required_files)
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "neuron_available": os.path.exists('/opt/aws/neuron'),
        "working_dir": WORKING_DIR,
        "max_execution_time": MAX_EXECUTION_TIME,
        "synthetic_dataset_available": synthetic_available,
        "dataset_cache_dir": DATASET_CACHE_DIR
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
            code = item.get('code', '')
            
            logger.info(f"Executing: {paper_title} ({paper_id})")
            
            # Wrap each paper execution in try-except to prevent one failure from breaking the batch
            try:
                if not code:
                    raise ValueError("Code is empty or missing")
                
                code_analysis = analyze_code(code)
                exec_result = execute_code(paper_id, code, timeout, paper_title=paper_title)
                exec_result.update(code_analysis)
                
                results[paper_id] = exec_result
                
            except Exception as e:
                logger.error(f"Error executing paper {paper_id}: {e}")
                logger.error(traceback.format_exc())
                # Return error result for this paper, but continue with others
                results[paper_id] = {
                    "success": False,
                    "execution_time": 0,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "error_message": f"Execution error: {str(e)}",
                    "error_type": "execution_error",
                    "timeout": False
                }

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

