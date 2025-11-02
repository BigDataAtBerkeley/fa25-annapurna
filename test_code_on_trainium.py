#!/usr/bin/env python3
"""
Test Generated Code on Trainium

This script tests a specific paper's generated code on Trainium without running
the full pipeline. It:
1. Downloads code from S3 (or uses local file)
2. Sends to Trainium for execution
3. Saves results to OpenSearch and S3
4. Displays metrics

Usage:
    # Test by paper ID (downloads from S3)
    python test_code_on_trainium.py --paper-id <paper_id>
    
    # Test by local file
    python test_code_on_trainium.py --file generated_code/my_code.py --paper-id <paper_id>
    
    # Test with custom timeout
    python test_code_on_trainium.py --paper-id <paper_id> --timeout 900
"""

import os
import sys
import boto3
import requests
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

load_dotenv()

# AWS Clients
session = boto3.Session(region_name='us-east-1')
s3_client = boto3.client('s3')
ec2_client = boto3.client('ec2', region_name='us-east-2')

# Configuration
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'research-papers-v2')
CODE_BUCKET = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
OUTPUTS_BUCKET = os.getenv('OUTPUTS_BUCKET', 'papers-test-outputs')
TRAINIUM_ENDPOINT = os.getenv('TRAINIUM_ENDPOINT')
TRAINIUM_INSTANCE_ID = os.getenv('TRAINIUM_INSTANCE_ID')
DEFAULT_TIMEOUT = 600  # 10 minutes

# OpenSearch setup
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, 'us-east-1', 'es')

os_client = OpenSearch(
    hosts=[{'host': OPENSEARCH_ENDPOINT.replace('https://', '').replace('http://', ''), 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)


def ensure_trainium_running() -> bool:
    """Ensure Trainium instance is running"""
    if not TRAINIUM_INSTANCE_ID:
        print("⚠️  No TRAINIUM_INSTANCE_ID set, assuming instance is already running")
        return True
    
    try:
        response = ec2_client.describe_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
        state = response['Reservations'][0]['Instances'][0]['State']['Name']
        
        if state == 'running':
            print(f"✓ Trainium instance {TRAINIUM_INSTANCE_ID} is running")
            return True
        elif state == 'stopped':
            print(f"⏳ Starting Trainium instance {TRAINIUM_INSTANCE_ID}...")
            ec2_client.start_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
            # Wait for instance to be running
            waiter = ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
            print("⏳ Waiting 60 seconds for Trainium services to start...")
            time.sleep(60)
            print(f"✓ Trainium instance {TRAINIUM_INSTANCE_ID} is now running")
            return True
        else:
            print(f"⚠️  Trainium instance {TRAINIUM_INSTANCE_ID} is in state: {state}")
            return False
    except Exception as e:
        print(f"❌ Error checking/starting Trainium instance: {e}")
        return False


def get_paper_info(paper_id: str) -> Dict[str, Any]:
    """Get paper info from OpenSearch"""
    try:
        response = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
        return response['_source']
    except Exception as e:
        print(f"⚠️  Could not fetch paper from OpenSearch: {e}")
        return {'title': 'Unknown Paper'}


def download_code_from_s3(paper_id: str) -> str:
    """Download generated code from S3"""
    try:
        # Try to find the code file in S3
        response = s3_client.list_objects_v2(
            Bucket=CODE_BUCKET,
            Prefix=f"{paper_id}/"
        )
        
        if 'Contents' not in response:
            raise FileNotFoundError(f"No code found for paper {paper_id} in S3")
        
        # Find the .py file
        code_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.py')]
        
        if not code_files:
            raise FileNotFoundError(f"No .py files found for paper {paper_id}")
        
        code_key = code_files[0]
        print(f"📥 Downloading code from s3://{CODE_BUCKET}/{code_key}")
        
        obj = s3_client.get_object(Bucket=CODE_BUCKET, Key=code_key)
        code = obj['Body'].read().decode('utf-8')
        
        return code, code_key
        
    except Exception as e:
        raise Exception(f"Failed to download code from S3: {e}")


def read_code_from_file(filepath: str) -> str:
    """Read code from local file"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        print(f"📄 Read code from {filepath}")
        return code
    except Exception as e:
        raise Exception(f"Failed to read code from file: {e}")


def send_to_trainium(paper_id: str, paper_title: str, code: str, timeout: int) -> Dict[str, Any]:
    """Send code to Trainium for execution"""
    if not TRAINIUM_ENDPOINT:
        raise ValueError("TRAINIUM_ENDPOINT environment variable not set")
    
    # Ensure Trainium is running
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
    
    print(f"\n🚀 Sending code to Trainium at {TRAINIUM_ENDPOINT}")
    print(f"   Timeout: {timeout}s")
    
    try:
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
        print(f"❌ Trainium execution timed out after {timeout} seconds")
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
    except Exception as e:
        print(f"❌ Error communicating with Trainium: {e}")
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


def save_outputs_to_s3(paper_id: str, exec_result: Dict[str, Any]):
    """Save execution outputs to S3"""
    print(f"\n💾 Saving outputs to S3...")
    
    saved_keys = []
    
    # Save stdout
    if exec_result.get('stdout'):
        key = f"{paper_id}/outputs/stdout.log"
        s3_client.put_object(
            Bucket=OUTPUTS_BUCKET,
            Key=key,
            Body=exec_result['stdout'].encode('utf-8')
        )
        saved_keys.append(key)
        print(f"   ✓ s3://{OUTPUTS_BUCKET}/{key}")
    
    # Save stderr
    if exec_result.get('stderr'):
        key = f"{paper_id}/outputs/stderr.log"
        s3_client.put_object(
            Bucket=OUTPUTS_BUCKET,
            Key=key,
            Body=exec_result['stderr'].encode('utf-8')
        )
        saved_keys.append(key)
        print(f"   ✓ s3://{OUTPUTS_BUCKET}/{key}")
    
    # Save metrics as JSON
    metrics = {k: v for k, v in exec_result.items() if k not in ['stdout', 'stderr']}
    key = f"{paper_id}/outputs/metrics.json"
    s3_client.put_object(
        Bucket=OUTPUTS_BUCKET,
        Key=key,
        Body=json.dumps(metrics, indent=2).encode('utf-8')
    )
    saved_keys.append(key)
    print(f"   ✓ s3://{OUTPUTS_BUCKET}/{key}")
    
    return saved_keys


def update_opensearch(paper_id: str, exec_result: Dict[str, Any], s3_keys: list):
    """Update OpenSearch with test results"""
    execution_hours = exec_result.get('execution_time', 0) / 3600
    estimated_cost = execution_hours * 1.34
    
    test_results = {
        # Core Test Status
        "tested": True,
        "tested_at": datetime.now().isoformat(),
        "test_success": exec_result.get('success', False),
        "test_in_progress": False,
        
        # Execution Metrics
        "execution_time": exec_result.get('execution_time', 0),
        "return_code": exec_result.get('return_code', -1),
        "timeout": exec_result.get('timeout', False),
        "executed_on": "trainium",
        "instance_type": "trn1.2xlarge",
        
        # Performance Metrics
        "peak_memory_mb": exec_result.get('peak_memory_mb'),
        "lines_of_code": exec_result.get('lines_of_code'),
        "has_training_loop": exec_result.get('has_training_loop'),
        "has_evaluation": exec_result.get('has_evaluation'),
        
        # S3 Artifact References
        "outputs_s3_bucket": OUTPUTS_BUCKET,
        "stdout_s3_key": f"{paper_id}/outputs/stdout.log",
        "stderr_s3_key": f"{paper_id}/outputs/stderr.log",
        "artifacts_s3_prefix": f"{paper_id}/outputs/",
        
        # Dataset Information
        "dataset_name": exec_result.get('dataset_name'),
        
        # Cost Tracking
        "trainium_hours": execution_hours,
        "estimated_compute_cost": round(estimated_cost, 4),
        
        # Error Info
        "has_errors": not exec_result.get('success', False),
        "error_message": exec_result.get('error_message') if not exec_result.get('success') else None,
        "error_type": exec_result.get('error_type') if not exec_result.get('success') else None,
        
        # Test metadata
        "test_attempts": 1,
        "last_test_attempt": datetime.now().isoformat()
    }
    
    # Remove None values
    test_results = {k: v for k, v in test_results.items() if v is not None}
    
    try:
        os_client.update(
            index=OPENSEARCH_INDEX,
            id=paper_id,
            body={"doc": test_results}
        )
        print(f"\n✓ Updated OpenSearch document {paper_id}")
    except Exception as e:
        print(f"\n⚠️  Failed to update OpenSearch: {e}")


def display_results(exec_result: Dict[str, Any]):
    """Display execution results"""
    print("\n" + "=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)
    
    # Status
    status = "✅ SUCCESS" if exec_result.get('success') else "❌ FAILED"
    print(f"\nStatus: {status}")
    
    # Metrics
    print(f"\nExecution Time: {exec_result.get('execution_time', 0):.2f}s")
    print(f"Return Code: {exec_result.get('return_code', -1)}")
    print(f"Timed Out: {exec_result.get('timeout', False)}")
    
    if exec_result.get('lines_of_code'):
        print(f"Lines of Code: {exec_result.get('lines_of_code')}")
    if exec_result.get('has_training_loop'):
        print(f"Has Training Loop: {exec_result.get('has_training_loop')}")
    if exec_result.get('peak_memory_mb'):
        print(f"Peak Memory: {exec_result.get('peak_memory_mb'):.2f} MB")
    if exec_result.get('dataset_name'):
        print(f"Dataset Used: {exec_result.get('dataset_name')}")
    
    # Cost
    hours = exec_result.get('execution_time', 0) / 3600
    cost = hours * 1.34
    print(f"\nEstimated Cost: ${cost:.4f}")
    
    # Output preview
    if exec_result.get('stdout'):
        print("\n" + "-" * 80)
        print("STDOUT (last 50 lines):")
        print("-" * 80)
        lines = exec_result['stdout'].split('\n')
        for line in lines[-50:]:
            print(line)
    
    if exec_result.get('stderr') and not exec_result.get('success'):
        print("\n" + "-" * 80)
        print("STDERR:")
        print("-" * 80)
        print(exec_result['stderr'])
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Test generated code on Trainium',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test by paper ID (downloads from S3)
  python test_code_on_trainium.py --paper-id 6-j63JkBP8oloYi_8CJH
  
  # Test local file
  python test_code_on_trainium.py --file generated_code/my_code.py --paper-id test_123
  
  # Test with custom timeout
  python test_code_on_trainium.py --paper-id 6-j63JkBP8oloYi_8CJH --timeout 900
        """
    )
    
    parser.add_argument('--paper-id', required=True, help='Paper ID')
    parser.add_argument('--file', help='Local code file to test (instead of downloading from S3)')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Execution timeout in seconds (default: 600)')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results to S3 and OpenSearch')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST CODE ON TRAINIUM")
    print("=" * 80)
    print(f"\nPaper ID: {args.paper_id}")
    print(f"Timeout: {args.timeout}s")
    
    # Get paper info
    paper_info = get_paper_info(args.paper_id)
    paper_title = paper_info.get('title', 'Unknown Paper')
    print(f"Paper Title: {paper_title}")
    
    # Get code
    try:
        if args.file:
            code = read_code_from_file(args.file)
        else:
            code, s3_key = download_code_from_s3(args.paper_id)
        
        print(f"Code length: {len(code)} characters")
        
    except Exception as e:
        print(f"\n❌ Error loading code: {e}")
        sys.exit(1)
    
    # Send to Trainium
    try:
        exec_result = send_to_trainium(args.paper_id, paper_title, code, args.timeout)
    except Exception as e:
        print(f"\n❌ Error executing on Trainium: {e}")
        sys.exit(1)
    
    # Save results
    if not args.no_save:
        s3_keys = save_outputs_to_s3(args.paper_id, exec_result)
        update_opensearch(args.paper_id, exec_result, s3_keys)
    else:
        print("\n⚠️  Skipping save (--no-save flag)")
    
    # Display results
    display_results(exec_result)


if __name__ == '__main__':
    main()

