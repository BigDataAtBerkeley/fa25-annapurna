#!/usr/bin/env python3
"""
Test the complete pipeline using papers already stored in S3.
Flow: S3 → researchQueue → Judge → OpenSearch → code-evaluation → CodeGen → code-testing → Tester
"""

import boto3
import json
import time
from datetime import datetime

# Configuration
S3_BUCKET = 'llm-research-papers'
RESEARCH_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo'
NUM_PAPERS = 10  # Send 10 papers to trigger code generation

def list_papers_from_s3(bucket, limit=10):
    """List papers from S3 bucket"""
    s3 = boto3.client('s3')
    papers = []
    
    print(f"📦 Fetching papers from S3 bucket: {bucket}")
    
    try:
        # List objects in the bucket
        response = s3.list_objects_v2(Bucket=bucket, MaxKeys=limit * 3)
        
        if 'Contents' not in response:
            print("⚠️ No papers found in S3 bucket")
            return []
        
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip if not a PDF
            if not key.endswith('.pdf'):
                continue
            
            # Extract metadata from key (assuming format like: ICLR/2025/paper_name.pdf)
            parts = key.split('/')
            
            paper = {
                'title': parts[-1].replace('.pdf', '').replace('_', ' '),
                'abstract': f'Research paper from {parts[0] if len(parts) > 1 else "unknown conference"}',
                'authors': ['Unknown'],
                'date': datetime.now().strftime('%Y-%m-%d'),
                's3_bucket': bucket,
                's3_key': key
            }
            
            papers.append(paper)
            
            if len(papers) >= limit:
                break
        
        print(f"✅ Found {len(papers)} papers in S3")
        return papers
        
    except Exception as e:
        print(f"❌ Error listing S3 objects: {e}")
        return []


def send_papers_to_queue(papers, queue_url):
    """Send papers to researchQueue.fifo"""
    sqs = boto3.client('sqs')
    
    print(f"\n📤 Sending {len(papers)} papers to researchQueue.fifo...")
    
    for i, paper in enumerate(papers, 1):
        try:
            # Create unique message group ID and deduplication ID
            timestamp = int(time.time() * 1000)  # milliseconds
            message_group_id = f"test-{timestamp}-{i}"
            dedup_id = f"test-{timestamp}-{i}-dedup"
            
            response = sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(paper),
                MessageGroupId=message_group_id,
                MessageDeduplicationId=dedup_id
            )
            
            print(f"  {i}. ✅ {paper['title'][:60]}...")
            print(f"      S3: s3://{paper['s3_bucket']}/{paper['s3_key']}")
            
            # Small delay to avoid throttling
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  {i}. ❌ Error: {e}")
    
    print(f"\n✅ Sent {len(papers)} papers to researchQueue.fifo")


def check_queue_status():
    """Check status of all queues"""
    sqs = boto3.client('sqs')
    
    queues = {
        'researchQueue.fifo': 'https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo',
        'code-evaluation.fifo': 'https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo',
        'code-testing.fifo': 'https://sqs.us-east-1.amazonaws.com/478852001205/code-testing.fifo'
    }
    
    print("\n" + "="*60)
    print("📊 Queue Status")
    print("="*60)
    
    for name, url in queues.items():
        try:
            response = sqs.get_queue_attributes(
                QueueUrl=url,
                AttributeNames=['ApproximateNumberOfMessages']
            )
            count = response['Attributes']['ApproximateNumberOfMessages']
            print(f"  {name}: {count} messages")
        except Exception as e:
            print(f"  {name}: Error - {e}")


def main():
    print("="*60)
    print("🚀 Testing Pipeline from S3 → Code Testing")
    print("="*60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPipeline Flow:")
    print("  1. S3 → researchQueue.fifo")
    print("  2. PapersJudge → OpenSearch + code-evaluation.fifo")
    print("  3. (10 papers) → PapersCodeGenerator → code-testing.fifo")
    print("  4. (10 codes) → PapersCodeTester → Trainium")
    print()
    
    # Step 1: Get papers from S3
    papers = list_papers_from_s3(S3_BUCKET, limit=NUM_PAPERS)
    
    if not papers:
        print("\n❌ No papers found in S3. Cannot proceed.")
        return
    
    # Step 2: Send to researchQueue
    send_papers_to_queue(papers, RESEARCH_QUEUE_URL)
    
    # Step 3: Check queue status
    check_queue_status()
    
    # Instructions
    print("\n" + "="*60)
    print("📋 Next Steps")
    print("="*60)
    print("\n1. Monitor PapersJudge logs:")
    print("   aws logs tail /aws/lambda/PapersJudge --follow")
    print("\n2. Wait for papers to be indexed in OpenSearch (~30-60 seconds)")
    print("\n3. Check OpenSearch:")
    print("   python check_opensearch.py | tail -30")
    print("\n4. Monitor code-evaluation queue:")
    print("   aws sqs get-queue-attributes \\")
    print("     --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo \\")
    print("     --attribute-names ApproximateNumberOfMessages")
    print("\n5. Once 10 papers accumulate, watch code generation:")
    print("   aws logs tail /aws/lambda/PapersCodeGenerator --follow")
    print("\n6. Check generated code in OpenSearch:")
    print("   python -c \"")
    print("from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth")
    print("import boto3, os")
    print("from dotenv import load_dotenv")
    print("load_dotenv()")
    print("session = boto3.Session(region_name='us-east-1')")
    print("creds = session.get_credentials().get_frozen_credentials()")
    print("auth = AWSV4SignerAuth(creds, 'us-east-1', 'es')")
    print("os_client = OpenSearch(")
    print("    hosts=[{'host': os.getenv('OPENSEARCH_ENDPOINT'), 'port': 443}],")
    print("    http_auth=auth, use_ssl=True, verify_certs=True,")
    print("    connection_class=RequestsHttpConnection")
    print(")")
    print("result = os_client.search(index='research-papers-v2', body={'query': {'term': {'code_generated': True}}, 'size': 20})")
    print("print(f'Papers with code: {result[\\\"hits\\\"][\\\"total\\\"][\\\"value\\\"]}')")
    print("\"")
    print("\n7. Monitor code testing:")
    print("   aws logs tail /aws/lambda/PapersCodeTester --follow")
    print("\n8. Check final test results:")
    print("   python check_opensearch.py | grep -E 'tested|test_success'")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

