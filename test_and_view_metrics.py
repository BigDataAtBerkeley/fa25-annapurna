#!/usr/bin/env python3
"""
Test Code on Trainium and View SageMaker Metrics

This script:
1. Tests a generated code file on Trainium
2. Waits for execution to complete
3. Queries and displays CloudWatch metrics that were logged
4. Shows how to view metrics in CloudWatch Console

Usage:
    # Test a paper by ID (downloads from S3)
    python test_and_view_metrics.py --paper-id <paper_id>
    
    # Test a local code file
    python test_and_view_metrics.py --file generated_code/my_code.py --paper-id <paper_id>
    
    # Test with custom timeout
    python test_and_view_metrics.py --paper-id <paper_id> --timeout 900
"""

import os
import sys
import boto3
import json
import argparse
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

load_dotenv()

# AWS Clients
session = boto3.Session(region_name='us-east-1')
s3_client = boto3.client('s3')
# Trainium region (us-east-2 or us-west-2 are supported for trn1 instances)
TRAINIUM_REGION = os.getenv('TRAINIUM_REGION', 'us-east-2')
ec2_client = boto3.client('ec2', region_name=TRAINIUM_REGION)
cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')

# Configuration
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'research-papers-v2')
CODE_BUCKET = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
OUTPUTS_BUCKET = os.getenv('OUTPUTS_BUCKET', 'papers-test-outputs')
TRAINIUM_ENDPOINT = os.getenv('TRAINIUM_ENDPOINT')
TRAINIUM_INSTANCE_ID = os.getenv('TRAINIUM_INSTANCE_ID')
DEFAULT_TIMEOUT = 600
METRICS_NAMESPACE = "Trainium/Training"

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
) if OPENSEARCH_ENDPOINT else None


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
            waiter = ec2_client.get_waiter('instance_running')
            waiter.wait(InstanceIds=[TRAINIUM_INSTANCE_ID])
            print("⏳ Waiting 60 seconds for Trainium services to start...")
            time.sleep(60)
            return True
        else:
            print(f"⚠️  Trainium instance is in state: {state}")
            return False
    except Exception as e:
        error_str = str(e)
        if 'InvalidInstanceID.NotFound' in error_str or 'does not exist' in error_str:
            print(f"⚠️  Instance ID {TRAINIUM_INSTANCE_ID} not found. Assuming instance is accessible via TRAINIUM_ENDPOINT")
        else:
            print(f"⚠️  Error checking Trainium instance: {e}. Assuming instance is running and accessible...")
        return True


def get_paper_info(paper_id: str) -> Dict[str, Any]:
    """Get paper info from OpenSearch"""
    if not os_client:
        return {'title': 'Unknown Paper'}
    try:
        response = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
        return response['_source']
    except Exception:
        return {'title': 'Unknown Paper'}


def download_code_from_s3(paper_id: str) -> tuple:
    """Download generated code from S3"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=CODE_BUCKET,
            Prefix=f"{paper_id}/"
        )
        
        if 'Contents' not in response:
            raise FileNotFoundError(f"No code found for paper {paper_id} in S3")
        
        code_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.py')]
        if not code_files:
            raise FileNotFoundError(f"No .py files found for paper {paper_id}")
        
        code_key = code_files[0]
        obj = s3_client.get_object(Bucket=CODE_BUCKET, Key=code_key)
        code = obj['Body'].read().decode('utf-8')
        return code, code_key
    except Exception as e:
        raise Exception(f"Failed to download code from S3: {e}")


def read_code_from_file(filepath: str) -> str:
    """Read code from local file"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Failed to read code from file: {e}")


def send_to_trainium(paper_id: str, paper_title: str, code: str, timeout: int) -> Dict[str, Any]:
    """Send code to Trainium for execution"""
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
    
    print(f"📡 Connecting to Trainium at {TRAINIUM_ENDPOINT}...")
    
    try:
        # First, try a quick health check
        try:
            health_response = requests.get(f"{TRAINIUM_ENDPOINT}/health", timeout=5)
            if health_response.status_code == 200:
                print("✓ Trainium health check passed")
            else:
                print(f"⚠️  Trainium health check returned status {health_response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not reach Trainium health endpoint: {e}")
            print("   Proceeding anyway...")
        
        print(f"📤 Sending code to Trainium (this may take a while)...")
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
        print(f"❌ Request timed out after {timeout + 30} seconds")
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
        print(f"❌ Could not connect to Trainium at {TRAINIUM_ENDPOINT}")
        print(f"   Error: {e}")
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


def save_outputs_to_s3(paper_id: str, exec_result: Dict[str, Any]) -> list:
    """Save execution outputs to S3"""
    saved_keys = []
    
    if exec_result.get('stdout'):
        key = f"{paper_id}/outputs/stdout.log"
        s3_client.put_object(Bucket=OUTPUTS_BUCKET, Key=key, Body=exec_result['stdout'].encode('utf-8'))
        saved_keys.append(key)
    
    if exec_result.get('stderr'):
        key = f"{paper_id}/outputs/stderr.log"
        s3_client.put_object(Bucket=OUTPUTS_BUCKET, Key=key, Body=exec_result['stderr'].encode('utf-8'))
        saved_keys.append(key)
    
    metrics = {k: v for k, v in exec_result.items() if k not in ['stdout', 'stderr']}
    key = f"{paper_id}/outputs/metrics.json"
    s3_client.put_object(Bucket=OUTPUTS_BUCKET, Key=key, Body=json.dumps(metrics, indent=2).encode('utf-8'))
    saved_keys.append(key)
    
    return saved_keys


def update_opensearch(paper_id: str, exec_result: Dict[str, Any], s3_keys: list):
    """Update OpenSearch with test results"""
    if not os_client:
        return
    
    execution_hours = exec_result.get('execution_time', 0) / 3600
    estimated_cost = execution_hours * 1.34
    
    test_results = {
        "tested": True,
        "tested_at": datetime.now().isoformat(),
        "test_success": exec_result.get('success', False),
        "test_in_progress": False,
        "execution_time": exec_result.get('execution_time', 0),
        "return_code": exec_result.get('return_code', -1),
        "timeout": exec_result.get('timeout', False),
        "executed_on": "trainium",
        "instance_type": "trn1.2xlarge",
        "peak_memory_mb": exec_result.get('peak_memory_mb'),
        "outputs_s3_bucket": OUTPUTS_BUCKET,
        "stdout_s3_key": f"{paper_id}/outputs/stdout.log",
        "stderr_s3_key": f"{paper_id}/outputs/stderr.log",
        "artifacts_s3_prefix": f"{paper_id}/outputs/",
        "trainium_hours": execution_hours,
        "estimated_compute_cost": round(estimated_cost, 4),
        "has_errors": not exec_result.get('success', False),
        "error_message": exec_result.get('error_message') if not exec_result.get('success') else None,
        "error_type": exec_result.get('error_type') if not exec_result.get('success') else None,
        "test_attempts": 1,
        "last_test_attempt": datetime.now().isoformat()
    }
    
    test_results = {k: v for k, v in test_results.items() if v is not None}
    
    try:
        os_client.update(index=OPENSEARCH_INDEX, id=paper_id, body={"doc": test_results})
    except Exception as e:
        print(f"⚠️  Failed to update OpenSearch: {e}")


def display_results(exec_result: Dict[str, Any]):
    """Display execution results"""
    print("\n" + "=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)
    
    status = "✅ SUCCESS" if exec_result.get('success') else "❌ FAILED"
    print(f"\nStatus: {status}")
    print(f"Execution Time: {exec_result.get('execution_time', 0):.2f}s")
    print(f"Return Code: {exec_result.get('return_code', -1)}")
    
    if exec_result.get('lines_of_code'):
        print(f"Lines of Code: {exec_result.get('lines_of_code')}")
    if exec_result.get('peak_memory_mb'):
        print(f"Peak Memory: {exec_result.get('peak_memory_mb'):.2f} MB")
    
    hours = exec_result.get('execution_time', 0) / 3600
    cost = hours * 1.34
    print(f"Estimated Cost: ${cost:.4f}")
    
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


def wait_for_metrics(paper_id: str, max_wait: int = 60) -> bool:
    """
    Wait for metrics to appear in CloudWatch.
    
    Args:
        paper_id: Paper ID to check for
        max_wait: Maximum seconds to wait
        
    Returns:
        True if metrics found, False otherwise
    """
    print(f"\n⏳ Waiting up to {max_wait}s for metrics to appear in CloudWatch...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = cloudwatch.list_metrics(
                Namespace=METRICS_NAMESPACE,
                Dimensions=[
                    {
                        'Name': 'PaperId',
                        'Value': paper_id
                    }
                ]
            )
            
            if response['Metrics']:
                print(f"✓ Found {len(response['Metrics'])} metrics in CloudWatch")
                return True
                
        except Exception as e:
            print(f"⚠️  Error checking for metrics: {e}")
        
        time.sleep(5)
    
    print("⚠️  No metrics found after waiting")
    return False


def list_metrics_for_paper(paper_id: str):
    """List all metrics for a specific paper"""
    try:
        response = cloudwatch.list_metrics(
            Namespace=METRICS_NAMESPACE,
            Dimensions=[
                {
                    'Name': 'PaperId',
                    'Value': paper_id
                }
            ]
        )
        
        if not response['Metrics']:
            print(f"\n❌ No metrics found for paper {paper_id}")
            print("   This could mean:")
            print("   1. Metrics haven't been logged yet (wait a few seconds)")
            print("   2. SAGEMAKER_METRICS_ENABLED is disabled")
            print("   3. IAM permissions missing (cloudwatch:PutMetricData)")
            print("   4. Code didn't output metrics in METRICS: format")
            return []
        
        metrics = response['Metrics']
        print(f"\n📊 Found {len(metrics)} metrics for paper {paper_id}:")
        print("=" * 80)
        
        # Group by metric name
        metric_names = {}
        for metric in metrics:
            name = metric['MetricName']
            if name not in metric_names:
                metric_names[name] = []
            metric_names[name].append(metric)
        
        for metric_name, metric_list in sorted(metric_names.items()):
            print(f"\n  📈 {metric_name}")
            for metric in metric_list:
                dimensions = {d['Name']: d['Value'] for d in metric.get('Dimensions', [])}
                if 'Step' in dimensions:
                    print(f"      Step: {dimensions['Step']}")
                if 'TrainingJobName' in dimensions:
                    print(f"      Job: {dimensions['TrainingJobName']}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ Error listing metrics: {e}")
        return []


def get_metric_statistics(paper_id: str, metric_name: str, hours_back: int = 1):
    """Get statistics for a specific metric"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        response = cloudwatch.get_metric_statistics(
            Namespace=METRICS_NAMESPACE,
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'PaperId',
                    'Value': paper_id
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5 minute periods
            Statistics=['Average', 'Maximum', 'Minimum', 'Sum']
        )
        
        if not response['Datapoints']:
            return None
        
        # Sort by timestamp
        datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
        
        print(f"\n📊 Statistics for '{metric_name}':")
        print("=" * 80)
        
        for dp in datapoints:
            timestamp = dp['Timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f"  {timestamp}:")
            print(f"    Average: {dp.get('Average', 'N/A')}")
            print(f"    Maximum: {dp.get('Maximum', 'N/A')}")
            print(f"    Minimum: {dp.get('Minimum', 'N/A')}")
            if 'Sum' in dp:
                print(f"    Sum: {dp['Sum']}")
        
        return datapoints
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return None


def display_metrics_summary(paper_id: str):
    """Display a summary of all metrics for the paper"""
    metrics = list_metrics_for_paper(paper_id)
    
    if not metrics:
        return
    
    # Get statistics for key metrics
    key_metrics = [
        'training_loss', 'training_accuracy', 'validation_accuracy', 
        'test_accuracy', 'execution_time_seconds', 'execution_success',
        'peak_memory_mb', 'estimated_cost_usd'
    ]
    
    print("\n" + "=" * 80)
    print("METRIC STATISTICS")
    print("=" * 80)
    
    for metric_name in key_metrics:
        metric_exists = any(m['MetricName'] == metric_name for m in metrics)
        if metric_exists:
            get_metric_statistics(paper_id, metric_name, hours_back=1)
    
    print("\n" + "=" * 80)
    print("HOW TO VIEW IN CLOUDWATCH CONSOLE:")
    print("=" * 80)
    print(f"""
1. Open AWS Console: https://console.aws.amazon.com/cloudwatch/
2. Navigate to: Metrics → All metrics
3. Select namespace: "{METRICS_NAMESPACE}"
4. Filter by dimension: PaperId = {paper_id}
5. Select metrics to visualize

Or use AWS CLI:
    aws cloudwatch list-metrics \\
        --namespace "{METRICS_NAMESPACE}" \\
        --dimensions Name=PaperId,Value={paper_id}
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Test code on Trainium and view SageMaker metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test by paper ID (downloads from S3)
  python test_and_view_metrics.py --paper-id 6-j63JkBP8oloYi_8CJH
  
  # Test local file
  python test_and_view_metrics.py --file generated_code/TurboAttention_MODIFIED_with_dataset_loader.py --paper-id test_123
  
  # Test with custom timeout
  python test_and_view_metrics.py --paper-id 6-j63JkBP8oloYi_8CJH --timeout 900
        """
    )
    
    parser.add_argument('--paper-id', required=True, help='Paper ID')
    parser.add_argument('--file', help='Local code file to test (instead of downloading from S3)')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='Execution timeout in seconds (default: 600)')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results to S3 and OpenSearch')
    parser.add_argument('--skip-execution', action='store_true', help='Skip execution, only view existing metrics')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST CODE ON TRAINIUM & VIEW SAGEMAKER METRICS")
    print("=" * 80)
    print(f"\nPaper ID: {args.paper_id}")
    
    # Get paper info
    paper_info = get_paper_info(args.paper_id)
    paper_title = paper_info.get('title', 'Unknown Paper')
    print(f"Paper Title: {paper_title}")
    
    if not args.skip_execution:
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
        print(f"\n🚀 Executing code on Trainium (timeout: {args.timeout}s)...")
        try:
            exec_result = send_to_trainium(args.paper_id, paper_title, code, args.timeout)
        except Exception as e:
            print(f"\n❌ Error executing on Trainium: {e}")
            sys.exit(1)
        
        # Save results
        if not args.no_save:
            s3_keys = save_outputs_to_s3(args.paper_id, exec_result)
            update_opensearch(args.paper_id, exec_result, s3_keys)
        
        # Display execution results
        display_results(exec_result)
        
        # Wait for metrics to be logged
        print("\n" + "=" * 80)
        print("CHECKING SAGEMAKER METRICS")
        print("=" * 80)
        wait_for_metrics(args.paper_id, max_wait=60)
    else:
        print("\n⏭️  Skipping execution (--skip-execution flag)")
    
    # Display metrics
    display_metrics_summary(args.paper_id)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

