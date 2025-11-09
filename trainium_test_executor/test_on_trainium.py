"""
Test generated code files on Trainium instance.

This script:
1. Reads Python code files from generated_code directory
2. Sends them to Trainium for execution
3. Saves results locally to trainium_test_results/ (no S3/OpenSearch)
4. Handles starting Trainium instance if needed (keeps it running by default)
5. Use --stop flag to stop the instance after testing
"""

import os
import sys
import json
import argparse
import logging
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
def safe_getenv(key: str, default: str = None, type_func=str):
    """Safely get environment variable with type conversion and cleaning."""
    value = os.getenv(key, default)
    if value is None:
        return None
    # Strip whitespace and take only the first part (in case of malformed .env)
    value = value.strip().split()[0] if value.strip() else default
    if value is None:
        return None
    try:
        return type_func(value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {key}: {value}, using default: {default}")
        return type_func(default) if default else None

TRAINIUM_ENDPOINT = safe_getenv('TRAINIUM_ENDPOINT')
TRAINIUM_INSTANCE_ID = safe_getenv('TRAINIUM_INSTANCE_ID')
TRAINIUM_REGION = safe_getenv('TRAINIUM_REGION', 'us-east-2')
TRAINIUM_TIMEOUT = safe_getenv('TRAINIUM_TIMEOUT', '1800', int)  # 30 minutes (increased for Neuron compilation)
GENERATED_CODE_DIR = safe_getenv('GENERATED_CODE_DIR', 'generated_code')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'trainium_test_results')

# AWS clients
ec2_client = boto3.client('ec2', region_name=TRAINIUM_REGION) if TRAINIUM_INSTANCE_ID else None


def get_instance_ip() -> Optional[str]:
    """
    Get the current public IP of the Trainium instance.
    Checks for Elastic IP first (static), then falls back to public IP.
    """
    if not TRAINIUM_INSTANCE_ID or not ec2_client:
        return None
    
    try:
        response = ec2_client.describe_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
        instance = response['Reservations'][0]['Instances'][0]
        
        # Check for Elastic IP first (static IP that doesn't change)
        network_interfaces = instance.get('NetworkInterfaces', [])
        for ni in network_interfaces:
            association = ni.get('Association', {})
            if association and association.get('PublicIp'):
                elastic_ip = association.get('PublicIp')
                logger.info(f"Found Elastic IP: {elastic_ip}")
                return elastic_ip
        
        # Fall back to regular public IP
        public_ip = instance.get('PublicIpAddress')
        if public_ip:
            logger.info(f"Found public IP: {public_ip} (not Elastic IP - may change on restart)")
        return public_ip
    except Exception as e:
        logger.warning(f"Could not get instance IP: {e}")
        return None


def ensure_trainium_running() -> bool:
    """
    Ensure Trainium instance is running (if TRAINIUM_INSTANCE_ID is set).
    Returns True if instance is running, False otherwise.
    """
    if not TRAINIUM_INSTANCE_ID:
        logger.info("No TRAINIUM_INSTANCE_ID set, assuming instance is already running")
        return True
    
    if not ec2_client:
        logger.warning("EC2 client not available, assuming instance is running")
        return True
    
    try:
        response = ec2_client.describe_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
        instance = response['Reservations'][0]['Instances'][0]
        state = instance['State']['Name']
        
        if state == 'running':
            logger.info(f"✓ Trainium instance {TRAINIUM_INSTANCE_ID} is running")
            # Check if IP has changed (check Elastic IP first, then public IP)
            current_ip = get_instance_ip()  # This checks Elastic IP first
            if current_ip and TRAINIUM_ENDPOINT:
                # Extract IP from endpoint
                endpoint_ip = TRAINIUM_ENDPOINT.replace('http://', '').replace('https://', '').split(':')[0]
                if current_ip != endpoint_ip:
                    logger.warning(f"⚠️ Instance IP ({current_ip}) differs from TRAINIUM_ENDPOINT ({endpoint_ip})")
                    logger.warning(f"   If using Elastic IP, this shouldn't happen. Check your .env file.")
                    logger.warning(f"   Update .env: TRAINIUM_ENDPOINT=http://{current_ip}:8000")
            return True
        elif state == 'stopped':
            logger.info(f"⏳ Starting Trainium instance {TRAINIUM_INSTANCE_ID}...")
            ec2_client.start_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
            # Wait for instance to be running
            waiter = ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
            # Get new IP address
            new_ip = get_instance_ip()
            if new_ip:
                # Check if this is an Elastic IP (shouldn't change)
                response = ec2_client.describe_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
                instance = response['Reservations'][0]['Instances'][0]
                network_interfaces = instance.get('NetworkInterfaces', [])
                has_elastic_ip = any(ni.get('Association', {}).get('PublicIp') for ni in network_interfaces)
                
                if has_elastic_ip:
                    logger.info(f"📌 Instance started with Elastic IP: {new_ip} (static, won't change)")
                else:
                    logger.info(f"📌 Instance started with new IP: {new_ip}")
                    logger.warning(f"⚠️ IP address changed! Consider using Elastic IP for static address.")
                    logger.warning(f"   Update .env: TRAINIUM_ENDPOINT=http://{new_ip}:8000")
            
            # Additional wait for services to start (Flask app, etc.)
            logger.info("⏳ Waiting 60 seconds for Trainium services to start...")
            time.sleep(60)
            logger.info(f"✓ Trainium instance {TRAINIUM_INSTANCE_ID} is now running")
            return True
        else:
            logger.warning(f"⚠️ Trainium instance {TRAINIUM_INSTANCE_ID} is in state: {state}")
            return False
            
    except Exception as e:
        error_str = str(e)
        if 'InvalidInstanceID.NotFound' in error_str or 'does not exist' in error_str:
            logger.warning(f"⚠️ Instance ID {TRAINIUM_INSTANCE_ID} not found. Assuming instance is accessible via TRAINIUM_ENDPOINT")
        else:
            logger.warning(f"⚠️ Error checking Trainium instance: {e}. Assuming instance is running and accessible...")
        return True


def stop_trainium_instance() -> bool:
    """Stop Trainium instance if TRAINIUM_INSTANCE_ID is set."""
    if not TRAINIUM_INSTANCE_ID or not ec2_client:
        return False
    
    try:
        response = ec2_client.describe_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
        state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
        if state == 'running':
            logger.info(f"🛑 Stopping Trainium instance {TRAINIUM_INSTANCE_ID}...")
            ec2_client.stop_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
            logger.info("✓ Stop command sent. Instance will stop shortly.")
            return True
        else:
            logger.info(f"Instance is already in state: {state}")
            return False
    except Exception as e:
        logger.error(f"Error stopping Trainium instance: {e}")
        return False


def get_code_files(code_dir: str) -> List[Path]:
    """Get all Python code files from generated_code directory."""
    code_path = Path(code_dir)
    
    if not code_path.exists():
        logger.error(f"Code directory does not exist: {code_dir}")
        return []
    
    # Find all .py files (excluding metadata files)
    code_files = [
        f for f in code_path.glob("*.py")
        if not f.name.endswith('_metadata.py')
    ]
    
    logger.info(f"Found {len(code_files)} code files in {code_dir}")
    return code_files


def read_code_file(file_path: Path) -> tuple[str, Dict[str, Any]]:
    """
    Read code file and extract metadata from header.
    
    Returns:
        Tuple of (code_content, metadata_dict)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to extract metadata from header
    metadata = {
        'filename': file_path.name,
        'paper_title': 'Unknown',
        'paper_id': 'unknown',
        'authors': [],
        'generated_at': datetime.now().isoformat()
    }
    
    # Try to read corresponding metadata JSON file
    # Metadata files are named: {filename}_metadata.json
    metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                json_metadata = json.load(f)
                metadata.update({
                    'paper_title': json_metadata.get('paper_title', 'Unknown'),
                    'paper_id': json_metadata.get('paper_id', 'unknown'),
                    'authors': json_metadata.get('paper_authors', []),
                    'generated_at': json_metadata.get('generated_at', metadata['generated_at'])
                })
        except Exception as e:
            logger.warning(f"Could not read metadata file {metadata_file}: {e}")
    
    # Also try to extract from docstring
    if content.startswith('"""'):
        try:
            docstring_end = content.find('"""', 3)
            if docstring_end > 0:
                docstring = content[3:docstring_end]
                for line in docstring.split('\n'):
                    if 'Paper ID:' in line:
                        metadata['paper_id'] = line.split('Paper ID:')[-1].strip()
                    if 'Paper Title:' in line or metadata['paper_title'] == 'Unknown':
                        if 'Paper Title:' in line:
                            metadata['paper_title'] = line.split('Paper Title:')[-1].strip()
        except:
            pass
    
    return content, metadata


def send_to_trainium(paper_id: str, paper_title: str, code: str, timeout: int) -> Dict[str, Any]:
    """Send code to Trainium for execution."""
    if not TRAINIUM_ENDPOINT:
        raise ValueError("TRAINIUM_ENDPOINT environment variable not set")
    
    if not ensure_trainium_running():
        raise RuntimeError("Failed to start Trainium instance")
    
    payload = {
        "batch": [{
            "paper_id": paper_id,
            "paper_title": paper_title,
            "code": code,
            "s3_code_key": f"{paper_id}/code.py"
        }],
        "timeout": timeout
    }
    
    logger.info(f"📡 Connecting to Trainium at {TRAINIUM_ENDPOINT}...")
    
    try:
        # Health check
        try:
            health_response = requests.get(f"{TRAINIUM_ENDPOINT}/health", timeout=5)
            if health_response.status_code == 200:
                logger.info("✓ Trainium health check passed")
            else:
                logger.warning(f"⚠️ Trainium health check returned status {health_response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Could not reach Trainium health endpoint: {e}")
            logger.info("   Proceeding anyway...")
        
        logger.info(f"📤 Sending code to Trainium (timeout: {timeout}s)...")
        response = requests.post(
            f"{TRAINIUM_ENDPOINT}/execute_batch",
            json=payload,
            timeout=timeout + 30
        )
        response.raise_for_status()
        results = response.json()
        
        if paper_id in results.get('results', {}):
            return results['results'][paper_id]
        else:
            raise Exception("No results returned from Trainium")
            
    except requests.exceptions.Timeout:
        logger.error(f"❌ Request timed out after {timeout + 30} seconds")
        return {
            "success": False,
            "execution_time": timeout,
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "timeout": True,
            "error_message": f"Execution timed out after {timeout} seconds",
            "error_type": "timeout"
        }
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Could not connect to Trainium at {TRAINIUM_ENDPOINT}")
        logger.error(f"   Error: {e}")
        return {
            "success": False,
            "execution_time": 0,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "timeout": False,
            "error_message": f"Could not connect to Trainium: {str(e)}",
            "error_type": "connection_error"
        }
    except Exception as e:
        logger.error(f"❌ Error communicating with Trainium: {e}")
        return {
            "success": False,
            "execution_time": 0,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "timeout": False,
            "error_message": f"Communication error: {str(e)}",
            "error_type": "trainium_error"
        }


def save_results_locally(paper_id: str, paper_title: str, exec_result: Dict[str, Any], 
                         code_content: str) -> Path:
    """
    Save execution results to local directory.
    
    Returns:
        Path to the results directory
    """
    # Create results directory structure
    results_path = Path(RESULTS_DIR) / paper_id
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save stdout
    if exec_result.get('stdout'):
        stdout_file = results_path / 'stdout.log'
        with open(stdout_file, 'w', encoding='utf-8') as f:
            f.write(exec_result['stdout'])
        logger.info(f"  ✓ Saved stdout to {stdout_file}")
    
    # Save stderr
    if exec_result.get('stderr'):
        stderr_file = results_path / 'stderr.log'
        with open(stderr_file, 'w', encoding='utf-8') as f:
            f.write(exec_result['stderr'])
        logger.info(f"  ✓ Saved stderr to {stderr_file}")
    
    # Save execution summary
    summary = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "success": exec_result.get('success', False),
        "execution_time": exec_result.get('execution_time', 0),
        "return_code": exec_result.get('return_code', -1),
        "timeout": exec_result.get('timeout', False),
        "error_message": exec_result.get('error_message'),
        "error_type": exec_result.get('error_type'),
        "tested_at": datetime.now().isoformat(),
        "detailed_metrics": exec_result.get('detailed_metrics', {})
    }
    
    summary_file = results_path / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Saved summary to {summary_file}")
    
    # Save code that was tested
    code_file = results_path / 'code.py'
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code_content)
    logger.info(f"  ✓ Saved code to {code_file}")
    
    # Save plots if available
    if exec_result.get('plots'):
        plots_dir = results_path / 'plots'
        plots_dir.mkdir(exist_ok=True)
        for plot_name, plot_data in exec_result.get('plots', {}).items():
            plot_file = plots_dir / plot_name
            # Assume plot_data is base64 encoded image
            import base64
            try:
                plot_bytes = base64.b64decode(plot_data)
                with open(plot_file, 'wb') as f:
                    f.write(plot_bytes)
                logger.info(f"  ✓ Saved plot to {plot_file}")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save plot {plot_name}: {e}")
    
    return results_path


def test_code_file(file_path: Path, timeout: int) -> Dict[str, Any]:
    """Test a single code file on Trainium."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {file_path.name}")
    logger.info(f"{'='*80}")
    
    # Read code and metadata
    code_content, metadata = read_code_file(file_path)
    paper_id = metadata.get('paper_id', file_path.stem)
    paper_title = metadata.get('paper_title', 'Unknown')
    
    logger.info(f"Paper ID: {paper_id}")
    logger.info(f"Paper Title: {paper_title}")
    
    # Send to Trainium
    try:
        exec_result = send_to_trainium(paper_id, paper_title, code_content, timeout)
        
        # Save results locally
        results_path = save_results_locally(paper_id, paper_title, exec_result, code_content)
        
        # Print summary
        status = "✓ SUCCESS" if exec_result.get('success') else "✗ FAILED"
        logger.info(f"\n{status}")
        logger.info(f"Execution time: {exec_result.get('execution_time', 0):.2f}s")
        logger.info(f"Return code: {exec_result.get('return_code', -1)}")
        logger.info(f"Results saved to: {results_path}")
        
        if not exec_result.get('success'):
            logger.error(f"Error: {exec_result.get('error_message', 'Unknown error')}")
            # Show detailed error output
            if exec_result.get('stderr'):
                logger.error(f"\n{'='*80}")
                logger.error("STDERR OUTPUT:")
                logger.error(f"{'='*80}")
                logger.error(exec_result.get('stderr'))
            if exec_result.get('stdout'):
                # Show last 30 lines of stdout for context
                stdout_lines = exec_result.get('stdout', '').split('\n')
                last_lines = '\n'.join(stdout_lines[-30:])
                logger.error(f"\n{'='*80}")
                logger.error("LAST 30 LINES OF STDOUT:")
                logger.error(f"{'='*80}")
                logger.error(last_lines)
        
        return {
            "file": str(file_path),
            "paper_id": paper_id,
            "paper_title": paper_title,
            "success": exec_result.get('success', False),
            "results_path": str(results_path),
            "exec_result": exec_result
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing {file_path.name}: {e}")
        return {
            "file": str(file_path),
            "paper_id": paper_id,
            "paper_title": paper_title,
            "success": False,
            "error": str(e)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test generated code files on Trainium instance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--code-dir', default=GENERATED_CODE_DIR,
                       help=f'Directory containing generated code files (default: {GENERATED_CODE_DIR})')
    parser.add_argument('--file', type=str,
                       help='Test a specific file (relative to code-dir)')
    parser.add_argument('--timeout', type=int, default=TRAINIUM_TIMEOUT,
                       help=f'Execution timeout in seconds (default: {TRAINIUM_TIMEOUT})')
    parser.add_argument('--stop', action='store_true',
                       help='Stop Trainium instance after testing (default: keeps instance running)')
    parser.add_argument('--stop-only', action='store_true',
                       help='Only stop Trainium instance (do not test)')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop testing after first connection error (useful when Trainium is unreachable)')
    
    args = parser.parse_args()
    
    # Handle stop-only mode
    if args.stop_only:
        if stop_trainium_instance():
            print("✓ Trainium instance stop command sent")
        else:
            print("⚠️ Could not stop instance (may not be set or already stopped)")
        return
    
    # Ensure Trainium is running
    if not ensure_trainium_running():
        print("❌ Failed to start Trainium instance. Exiting.")
        sys.exit(1)
    
    # Get code files
    if args.file:
        # Test specific file
        file_path = Path(args.code_dir) / args.file
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
        code_files = [file_path]
    else:
        # Test all files
        code_files = get_code_files(args.code_dir)
        if not code_files:
            print(f"❌ No code files found in {args.code_dir}")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"TRAINIUM CODE TESTING")
    print(f"{'='*80}")
    print(f"Code directory: {args.code_dir}")
    print(f"Files to test: {len(code_files)}")
    print(f"Timeout: {args.timeout}s")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"{'='*80}\n")
    
    # Test each file
    results = []
    connection_error_occurred = False
    
    for code_file in code_files:
        result = test_code_file(code_file, args.timeout)
        results.append(result)
        
        # Check if this was a connection error
        if args.stop_on_error:
            exec_result = result.get('exec_result', {})
            error_type = exec_result.get('error_type', '')
            if error_type in ['connection_error', 'trainium_error']:
                logger.error(f"\n❌ Connection error detected. Stopping testing (--stop-on-error flag).")
                logger.error(f"   Error: {exec_result.get('error_message', 'Unknown connection error')}")
                logger.error(f"   Fix the connection issue and try again.")
                connection_error_occurred = True
                break
    
    # Print summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r.get('success'))
    failed = len(results) - successful
    print(f"Total files tested: {len(results)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*80}")
    
    # Stop instance after testing (only if --stop flag is used)
    if args.stop and TRAINIUM_INSTANCE_ID:
        logger.info("\n🛑 Stopping Trainium instance after testing...")
        stop_trainium_instance()
    elif TRAINIUM_INSTANCE_ID:
        logger.info("\n✓ Trainium instance will remain running (use --stop to stop it)")
    elif args.no_stop:
        logger.info("\n⚠️ Instance will remain running (--no-stop was used)")
    
    # Exit with error code if any failed or connection error occurred
    if failed > 0 or connection_error_occurred:
        if connection_error_occurred:
            logger.error("\n💡 Tip: Check your TRAINIUM_ENDPOINT and ensure the Flask app is running on Trainium.")
            logger.error("   You can also use --stop-on-error to stop immediately on connection errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()

