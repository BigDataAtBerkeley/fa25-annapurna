#!/usr/bin/env python3
"""
Check if a paper was successfully executed on Trainium and display results.
"""

import sys
import boto3
import json
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import Dict, Any, Optional

def get_paper_execution_status(paper_id: str) -> Dict[str, Any]:
    """Get execution status for a paper from OpenSearch."""
    session = boto3.Session(region_name='us-east-1')
    creds = session.get_credentials().get_frozen_credentials()
    auth = AWSV4SignerAuth(creds, 'us-east-1', 'es')
    
    client = OpenSearch(
        hosts=[{'host': 'search-research-papers-uv3fxq76j5bkxq3bgp3nyfdtnm.us-east-1.es.amazonaws.com', 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    try:
        response = client.get(index='research-papers-v2', id=paper_id)
        paper = response['_source']
        
        return {
            'paper_id': paper_id,
            'paper_title': paper.get('title', 'Unknown'),
            'code_generated': paper.get('code_generated', False),
            'code_s3_bucket': paper.get('code_s3_bucket'),
            'code_s3_key': paper.get('code_s3_key'),
            'tested': paper.get('tested', False),
            'test_success': paper.get('test_success', False),
            'tested_at': paper.get('tested_at'),
            'execution_time': paper.get('execution_time'),
            'return_code': paper.get('return_code'),
            'timeout': paper.get('timeout', False),
            'error_message': paper.get('error_message'),
            'outputs_s3_bucket': paper.get('outputs_s3_bucket'),
            'stdout_s3_key': paper.get('stdout_s3_key'),
            'stderr_s3_key': paper.get('stderr_s3_key'),
            'training_loss': paper.get('training_loss'),
            'validation_accuracy': paper.get('validation_accuracy'),
            'peak_memory_mb': paper.get('peak_memory_mb'),
        }
    except Exception as e:
        return {'error': str(e)}

def download_s3_file(bucket: str, key: str) -> Optional[str]:
    """Download a file from S3 and return its content."""
    if not bucket or not key:
        return None
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        return f"Error downloading: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_trainium_execution.py <paper_id>")
        print("Example: python check_trainium_execution.py j-hHZpoBclM7MZc3vpPD")
        sys.exit(1)
    
    paper_id = sys.argv[1]
    
    print("=" * 80)
    print(f"TRAINIUM EXECUTION STATUS CHECK")
    print("=" * 80)
    print(f"Paper ID: {paper_id}")
    print()
    
    status = get_paper_execution_status(paper_id)
    
    if 'error' in status:
        print(f"❌ Error: {status['error']}")
        sys.exit(1)
    
    print(f"Paper Title: {status['paper_title']}")
    print()
    
    # Code Generation Status
    print("=" * 80)
    print("CODE GENERATION STATUS")
    print("=" * 80)
    if status['code_generated']:
        print("✅ Code Generated: Yes")
        print(f"   S3 Bucket: {status['code_s3_bucket']}")
        print(f"   S3 Key: {status['code_s3_key']}")
    else:
        print("❌ Code Generated: No")
    print()
    
    # Execution Status
    print("=" * 80)
    print("TRAINIUM EXECUTION STATUS")
    print("=" * 80)
    if status['tested']:
        print("✅ Tested: Yes")
        print(f"   Tested At: {status['tested_at']}")
        print(f"   Success: {'✅ Yes' if status['test_success'] else '❌ No'}")
        print(f"   Execution Time: {status['execution_time']}s" if status['execution_time'] else "   Execution Time: N/A")
        print(f"   Return Code: {status['return_code']}")
        print(f"   Timeout: {'Yes' if status['timeout'] else 'No'}")
        
        if status.get('error_message'):
            print(f"   Error: {status['error_message']}")
        
        print()
        print("   METRICS:")
        if status.get('training_loss') is not None:
            print(f"     Training Loss: {status['training_loss']}")
        if status.get('validation_accuracy') is not None:
            print(f"     Validation Accuracy: {status['validation_accuracy']}")
        if status.get('peak_memory_mb'):
            print(f"     Peak Memory: {status['peak_memory_mb']} MB")
        
        print()
        print("   S3 OUTPUTS:")
        if status['outputs_s3_bucket']:
            print(f"     Bucket: {status['outputs_s3_bucket']}")
            if status['stdout_s3_key']:
                print(f"     Stdout: s3://{status['outputs_s3_bucket']}/{status['stdout_s3_key']}")
            if status['stderr_s3_key']:
                print(f"     Stderr: s3://{status['outputs_s3_bucket']}/{status['stderr_s3_key']}")
        else:
            print("     No outputs stored in S3")
        
        # Download and show stdout/stderr if available
        if status['stdout_s3_key'] and status['outputs_s3_bucket']:
            print()
            print("   STDOUT (last 50 lines):")
            print("   " + "-" * 76)
            stdout = download_s3_file(status['outputs_s3_bucket'], status['stdout_s3_key'])
            if stdout:
                lines = stdout.split('\n')
                for line in lines[-50:]:
                    print(f"   {line}")
            
        if status['stderr_s3_key'] and status['outputs_s3_bucket']:
            print()
            print("   STDERR:")
            print("   " + "-" * 76)
            stderr = download_s3_file(status['outputs_s3_bucket'], status['stderr_s3_key'])
            if stderr:
                print(f"   {stderr}")
    else:
        print("⏳ Tested: No (not executed on Trainium yet)")
        print()
        print("   Possible reasons:")
        print("   - Code was just generated and is waiting in the testing queue")
        print("   - Testing Lambda is processing other papers")
        print("   - Testing Lambda encountered an error (check Lambda logs)")
    print()
    
    print("=" * 80)
    
    # Check queue status
    sqs_client = boto3.client('sqs', region_name='us-east-1')
    try:
        queue_attrs = sqs_client.get_queue_attributes(
            QueueUrl='https://sqs.us-east-1.amazonaws.com/478852001205/code-testing.fifo',
            AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
        )
        visible = queue_attrs['Attributes'].get('ApproximateNumberOfMessages', '0')
        in_flight = queue_attrs['Attributes'].get('ApproximateNumberOfMessagesNotVisible', '0')
        print(f"Testing Queue Status: {visible} visible, {in_flight} in flight")
    except Exception as e:
        print(f"Could not check queue status: {e}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

