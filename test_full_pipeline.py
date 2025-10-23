#!/usr/bin/env python3
"""
This tests the complete pipeline assuming we already have papers indexed in OpenSearch. So this takes documents from OpenSearch and sends them to our code-evaluation SQS. Once a paper hits the front of the queue, its triggered by our PapersCodeGenerator Lambda, which generates code for the paper and sends it to our code-testing SQS. Then, the PapersCodeTester Lambda waits for 10 papers to be in the queue before being triggered and sends all 10 generated code files to our Trainium instance for testing. Once the testing is complete, the metrics and test results are saved as fields in our OpenSearch index.
"""

import boto3
import json
import time
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

load_dotenv()

session = boto3.Session(region_name='us-east-1')
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, 'us-east-1', 'es')

os_client = OpenSearch(
    hosts=[{'host': os.getenv('OPENSEARCH_ENDPOINT'), 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

sqs = boto3.client('sqs', region_name='us-east-1')

# Queue URLs
CODE_EVAL_QUEUE = 'https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo' ## code-generation queue
CODE_TEST_QUEUE = 'https://sqs.us-east-1.amazonaws.com/478852001205/code-testing.fifo' ## code-testing queue


def get_papers_without_code(limit=10):
    """Get papers from OpenSearch that don't have generated code yet"""
    print(f"\n📊 Fetching papers that haven't had code generated for them yet. Limit (as defined in the method call) is {limit} papers")
    
    result = os_client.search(
        index='research-papers-v2',
        body={
            'query': {
                'bool': {
                    'must_not': [
                        {'exists': {'field': 'code_generated'}} ## code_generated is a field in our index. if that field is null, code hasn't been gernerated for it yet 
                    ]
                }
            },
            'size': limit
        }
    )
    
    papers = []
    for hit in result['hits']['hits']:
        papers.append({
            'paper_id': hit['_id'],
            'title': hit['_source'].get('title', 'Unknown'),
            'abstract': hit['_source'].get('abstract', ''),
            'url': hit['_source'].get('url', ''),
            'pdf_link': hit['_source'].get('pdf_link', '')
        })
    
    print(f"✅ Found {len(papers)} papers without code")
    return papers


def send_to_code_generation(papers):
    """Send papers to code-evaluation queue"""
    print(f"\nSending {len(papers)} papers to code-evaluation queue...")
    
    for i, paper in enumerate(papers, 1):
        message = {
            'paper_id': paper['paper_id'],
            'paper_title': paper['title'],
            'abstract': paper['abstract'],
            'url': paper['url'],
            'pdf_link': paper['pdf_link']
        }
        
        try:
            sqs.send_message(
                QueueUrl=CODE_EVAL_QUEUE,
                MessageBody=json.dumps(message),
                MessageGroupId='code-generation',
                MessageDeduplicationId=f"{paper['paper_id']}-{int(time.time())}"
            )
            print(f"  {i}.{paper['title'][:70]}")
        except Exception as e:
            print(f"  {i}. {paper['title'][:70]}: {e}")
    
    print(f"Sent {len(papers)} papers to code-evaluation queue")


def check_queue_status():
    """Check status of all queues"""
    print("\nQueue Status:")
    print("=" * 80)
    
    queues = {
        'code-evaluation.fifo': CODE_EVAL_QUEUE,
        'code-testing.fifo': CODE_TEST_QUEUE
    }
    
    for name, url in queues.items():
        try:
            attrs = sqs.get_queue_attributes(
                QueueUrl=url,
                AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
            )
            visible = attrs['Attributes'].get('ApproximateNumberOfMessages', '0')
            in_flight = attrs['Attributes'].get('ApproximateNumberOfMessagesNotVisible', '0')
            print(f"  {name:25} → {visible:>3} waiting, {in_flight:>3} in-flight")
        except Exception as e:
            print(f"  {name:25} → Error: {e}")


def monitor_code_generation():
    """Monitor the code generation Lambda"""
    
    logs_client = boto3.client('logs', region_name='us-east-1')
    log_group = '/aws/lambda/PapersCodeGenerator'
    
    try:
        # Stream logs
        import subprocess
        subprocess.run(
            ['aws', 'logs', 'tail', log_group, '--follow', '--since', '1m'],
            timeout=300  
        )
    except subprocess.TimeoutExpired:
        print("\nMonitoring timeout")
    except KeyboardInterrupt:
        print("\nSkipping monitoring")


def check_generated_code():
    """Check if code was generated"""
    print("\nChecking for generated code in OpenSearch...")
    
    result = os_client.search(
        index='research-papers-v2',
        body={
            'query': {
                'bool': {
                    'must': [
                        {'exists': {'field': 'code_generated'}},
                        {'term': {'code_generated': True}}
                    ]
                }
            },
            'size': 10,
            'sort': [{'code_generated_at': 'desc'}]
        }
    )
    
    papers = result['hits']['hits']
    print(f"Found {len(papers)} papers with generated code:")
    
    for i, hit in enumerate(papers, 1):
        paper = hit['_source']
        print(f"  {i}. {paper.get('title', 'Unknown')[:70]}")
        print(f"     Code: s3://{paper.get('code_s3_bucket')}/{paper.get('code_s3_key')}")
    
    return papers


def send_to_testing(papers):
    """Send papers with code to testing queue"""
    print(f"\nSending {len(papers)} papers to code-testing queue...")
    
    for i, hit in enumerate(papers, 1):
        paper = hit['_source']
        message = {
            'paper_id': hit['_id'],
            'paper_title': paper.get('title'),
            's3_bucket': paper.get('code_s3_bucket'),
            's3_code_key': paper.get('code_s3_key'),
            's3_metadata_key': paper.get('code_s3_key', '').replace('code.py', 'metadata.json')
        }
        
        try:
            sqs.send_message(
                QueueUrl=CODE_TEST_QUEUE,
                MessageBody=json.dumps(message),
                MessageGroupId='code-testing',
                MessageDeduplicationId=f"{hit['_id']}-{int(time.time())}"
            )
            print(f"  {i}. {paper.get('title', 'Unknown')[:70]}")
        except Exception as e:
            print(f"  {i}. Error: {e}")


def monitor_testing():
    """Monitor the code testing Lambda"""
    print("\nMonitoring PapersCodeTester Lambda...")
    print("Waiting for batch of 10 papers to trigger testing on Trainium...")
    print("(Press Ctrl+C to stop)")
    
    try:
        import subprocess
        subprocess.run(
            ['aws', 'logs', 'tail', '/aws/lambda/PapersCodeTester', '--follow', '--since', '1m']
        )
    except KeyboardInterrupt:
        print("\nStopped monitoring")


def check_test_results():
    """Check test results"""
    print("\nChecking for test results in OpenSearch...")
    
    result = os_client.search(
        index='research-papers-v2',
        body={
            'query': {
                'bool': {
                    'must': [
                        {'exists': {'field': 'tested'}},
                        {'term': {'tested': True}}
                    ]
                }
            },
            'size': 10,
            'sort': [{'tested_at': 'desc'}]
        }
    )
    
    papers = result['hits']['hits']
    print(f"Found {len(papers)} tested papers:")
    
    for i, hit in enumerate(papers, 1):
        paper = hit['_source']
        success = "✅" if paper.get('test_success') else "❌"
        print(f"  {i}. {success} {paper.get('title', 'Unknown')[:60]}")
        print(f"     Execution: {paper.get('execution_time', 0):.2f}s | Cost: ${paper.get('estimated_cost', 0):.4f}")
        if paper.get('stdout_s3_key'):
            print(f"     Outputs: s3://{paper.get('outputs_s3_bucket')}/{paper.get('stdout_s3_key')}")


def main():
    # Step 1: Get papers without code
    papers = get_papers_without_code(limit=10)
    
    if not papers:
        print("\n⚠️  No papers without code found. Checking for papers with code instead...")
        papers_with_code = check_generated_code()
        
        if papers_with_code:
            print("\nSkipping code generation, going straight to testing...")
            send_to_testing(papers_with_code[:10])
            check_queue_status()
            monitor_testing()
            check_test_results()
        else:
            print("\nNo papers found at all")
        return
    
    # Step 2: Send to code generation
    send_to_code_generation(papers)
    check_queue_status()
    
    # Step 3: Monitor code generation
    monitor_code_generation()
    
    # Step 4: Check generated code
    papers_with_code = check_generated_code()
    
    if not papers_with_code:
        print("\nNo code generated yet. Run this script again later or check Lambda logs.")
        return
    
    # Step 5: Send to testing
    send_to_testing(papers_with_code[:10])
    check_queue_status()
    
    # Step 6: Monitor testing
    monitor_testing()
    
    # Step 7: Check results
    check_test_results()
    
    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Download test results: python download_test_results.py")
    print("  2. Check OpenSearch: python check_opensearch.py")
    print("  3. View Lambda logs: aws logs tail /aws/lambda/PapersCodeTester --follow")


if __name__ == '__main__':
    main()

