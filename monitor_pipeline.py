#!/usr/bin/env python3
"""
Monitor the full pipeline: code generation → Trainium execution

This script:
1. Monitors papers sent to code-evaluation queue
2. Tracks code generation status
3. Tracks Trainium execution status
4. Displays comprehensive results for both stages

Usage:
    python monitor_pipeline.py [--paper-ids ID1 ID2 ID3]
    python monitor_pipeline.py  # Monitors all papers in recent queue activity
"""

import os
import sys
import json
import time
import logging
import boto3
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add code_gen to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code_gen'))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v2")

def get_opensearch_client():
    """Get OpenSearch client"""
    from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
    
    session = boto3.Session(region_name=AWS_REGION)
    creds = session.get_credentials().get_frozen_credentials()
    auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
    
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", ""), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )

def get_paper_status(paper_id: str, os_client) -> Dict[str, Any]:
    """Get comprehensive status for a paper from OpenSearch"""
    try:
        doc = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
        source = doc['_source']
        
        return {
            'paper_id': paper_id,
            'paper_title': source.get('title', 'Unknown'),
            'code_generated': source.get('code_generated', False),
            'code_generated_at': source.get('code_generated_at'),
            'code_generation_error': source.get('code_generation_error'),
            'code_s3_bucket': source.get('code_s3_bucket'),
            'code_s3_key': source.get('code_s3_key'),
            'recommended_dataset': source.get('recommended_dataset'),
            'tested': source.get('tested', False),
            'test_success': source.get('test_success', False),
            'tested_at': source.get('tested_at'),
            'execution_time': source.get('execution_time'),
            'return_code': source.get('return_code'),
            'timeout': source.get('timeout', False),
            'error_message': source.get('error_message'),
            'error_type': source.get('error_type'),
            'outputs_s3_bucket': source.get('outputs_s3_bucket'),
            'stdout_s3_key': source.get('stdout_s3_key'),
            'stderr_s3_key': source.get('stderr_s3_key'),
            'training_loss': source.get('training_loss'),
            'validation_accuracy': source.get('validation_accuracy'),
            'peak_memory_mb': source.get('peak_memory_mb'),
            'estimated_compute_cost': source.get('estimated_compute_cost'),
        }
    except Exception as e:
        logger.error(f"Error getting status for {paper_id}: {e}")
        return {'paper_id': paper_id, 'error': str(e)}

def print_code_gen_status(status: Dict[str, Any]):
    """Print code generation status"""
    paper_id = status['paper_id']
    title = status['paper_title']
    
    print(f"\n{'='*80}")
    print(f"📄 Paper: {title[:60]}...")
    print(f"   ID: {paper_id}")
    print(f"{'='*80}")
    
    # Code Generation Status
    print(f"\n📝 CODE GENERATION:")
    if status.get('code_generated'):
        print(f"   ✅ Status: SUCCESS")
        print(f"   📅 Generated at: {status.get('code_generated_at', 'N/A')}")
        print(f"   📦 S3: s3://{status.get('code_s3_bucket')}/{status.get('code_s3_key')}")
        print(f"   🗂️  Dataset: {status.get('recommended_dataset', 'N/A')}")
    elif status.get('code_generation_error'):
        print(f"   ❌ Status: FAILED")
        print(f"   ⚠️  Error: {status.get('code_generation_error')}")
    else:
        print(f"   ⏳ Status: IN PROGRESS or NOT STARTED")

def print_trainium_status(status: Dict[str, Any]):
    """Print Trainium execution status"""
    print(f"\n🚀 TRAINIUM EXECUTION:")
    
    if not status.get('tested'):
        print(f"   ⏳ Status: NOT TESTED YET")
        if status.get('code_generated'):
            print(f"   💡 Code generated, waiting for execution...")
        return
    
    if status.get('test_success'):
        print(f"   ✅ Status: SUCCESS")
        exec_time = status.get('execution_time') or 0
        print(f"   ⏱️  Execution time: {exec_time:.2f}s")
        print(f"   📊 Return code: {status.get('return_code', -1)}")
        
        if status.get('training_loss') is not None:
            print(f"   📈 Training loss: {status.get('training_loss')}")
        if status.get('validation_accuracy') is not None:
            print(f"   📈 Validation accuracy: {status.get('validation_accuracy')}")
        peak_mem = status.get('peak_memory_mb')
        if peak_mem is not None:
            print(f"   💾 Peak memory: {peak_mem:.1f} MB")
        cost = status.get('estimated_compute_cost')
        if cost is not None:
            print(f"   💰 Estimated cost: ${cost:.2f}")
        
        print(f"   📅 Tested at: {status.get('tested_at', 'N/A')}")
        if status.get('outputs_s3_bucket'):
            print(f"   📦 Outputs: s3://{status.get('outputs_s3_bucket')}/{status.get('paper_id')}/outputs/")
    else:
        print(f"   ❌ Status: FAILED")
        exec_time = status.get('execution_time') or 0
        print(f"   ⏱️  Execution time: {exec_time:.2f}s")
        print(f"   📊 Return code: {status.get('return_code', -1)}")
        if status.get('timeout'):
            print(f"   ⏰ Timeout: YES")
        if status.get('error_message'):
            print(f"   ⚠️  Error: {status.get('error_message')}")
        if status.get('error_type'):
            print(f"   🔍 Error type: {status.get('error_type')}")
        if status.get('stderr_s3_key'):
            print(f"   📦 Stderr: s3://{status.get('outputs_s3_bucket')}/{status.get('stderr_s3_key')}")

def monitor_papers(paper_ids: List[str], watch: bool = False, check_interval: int = 10):
    """Monitor papers through the pipeline"""
    os_client = get_opensearch_client()
    
    print(f"\n🔍 Monitoring {len(paper_ids)} paper(s)...")
    print(f"   Check interval: {check_interval}s")
    print(f"   Watch mode: {'ON' if watch else 'OFF'}")
    
    if watch:
        print(f"\n💡 Press Ctrl+C to stop monitoring\n")
    
    all_complete = False
    iteration = 0
    
    while not all_complete or watch:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"📊 Status Check #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        all_complete = True
        results = {}
        
        for paper_id in paper_ids:
            status = get_paper_status(paper_id, os_client)
            results[paper_id] = status
            
            print_code_gen_status(status)
            print_trainium_status(status)
            
            # Check if pipeline is complete for this paper
            code_done = status.get('code_generated') or status.get('code_generation_error')
            test_done = status.get('tested')
            
            if not (code_done and test_done):
                all_complete = False
        
        # Summary
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY:")
        code_gen_success = sum(1 for r in results.values() if r.get('code_generated'))
        code_gen_failed = sum(1 for r in results.values() if r.get('code_generation_error'))
        test_success = sum(1 for r in results.values() if r.get('test_success'))
        test_failed = sum(1 for r in results.values() if r.get('tested') and not r.get('test_success'))
        test_pending = sum(1 for r in results.values() if r.get('code_generated') and not r.get('tested'))
        
        print(f"   Code Generation: ✅ {code_gen_success} | ❌ {code_gen_failed} | ⏳ {len(paper_ids) - code_gen_success - code_gen_failed}")
        print(f"   Trainium Execution: ✅ {test_success} | ❌ {test_failed} | ⏳ {test_pending}")
        
        if all_complete:
            print(f"\n✅ All papers have completed the pipeline!")
            if not watch:
                break
        else:
            print(f"\n⏳ Pipeline still in progress...")
            if not watch:
                print(f"   Run with --watch to continuously monitor")
        
        if watch and not all_complete:
            print(f"\n⏳ Waiting {check_interval}s before next check...")
            time.sleep(check_interval)
        elif watch:
            print(f"\n⏳ All complete. Waiting {check_interval}s before next check...")
            time.sleep(check_interval)
        else:
            break
    
    return results

def get_recent_papers_from_queue(max_papers: int = 3) -> List[str]:
    """Get recent paper IDs from code-evaluation queue (approximate)"""
    try:
        sqs = boto3.client('sqs', region_name=AWS_REGION)
        account_id = boto3.client('sts').get_caller_identity()['Account']
        queue_url = f"https://sqs.{AWS_REGION}.amazonaws.com/{account_id}/code-evaluation.fifo"
        
        # Get approximate number of messages
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['ApproximateNumberOfMessages']
        )
        num_messages = int(attrs['Attributes'].get('ApproximateNumberOfMessages', '0'))
        
        if num_messages == 0:
            logger.info("No messages in code-evaluation queue")
            return []
        
        # Try to peek at messages (without consuming)
        # Note: This is approximate - we can't peek without consuming in SQS
        logger.info(f"Found ~{num_messages} messages in queue (cannot peek without consuming)")
        return []
        
    except Exception as e:
        logger.warning(f"Could not check queue: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Monitor pipeline: code generation → Trainium execution')
    parser.add_argument(
        '--paper-ids',
        nargs='+',
        help='Paper IDs to monitor (if not provided, will try to find recent papers)'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuously monitor until Ctrl+C'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Check interval in seconds (default: 10)'
    )
    args = parser.parse_args()
    
    paper_ids = args.paper_ids
    
    if not paper_ids:
        logger.info("No paper IDs provided. Checking for recent papers...")
        # Try to get recent papers (this is approximate)
        recent = get_recent_papers_from_queue()
        if recent:
            paper_ids = recent[:3]  # Limit to 3
            logger.info(f"Found {len(paper_ids)} recent papers to monitor")
        else:
            logger.error("❌ No paper IDs provided and none found in queue.")
            logger.info("💡 Usage: python monitor_pipeline.py --paper-ids ID1 ID2 ID3")
            logger.info("💡 Or run: python grab_papers_for_code_gen.py first")
            return 1
    
    try:
        results = monitor_papers(paper_ids, watch=args.watch, check_interval=args.interval)
        
        print(f"\n{'='*80}")
        print(f"✅ Monitoring complete!")
        print(f"{'='*80}")
        print(f"\n💡 To view detailed results for a specific paper:")
        print(f"   python check_trainium_execution.py <paper_id>")
        print(f"\n💡 To view CloudWatch logs:")
        print(f"   Code Gen: aws logs tail /aws/lambda/PapersCodeGenerator --follow")
        print(f"   Trainium: aws logs tail /aws/lambda/PapersCodeTester --follow")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Monitoring stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

