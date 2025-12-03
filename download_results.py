#!/usr/bin/env python3
"""
Download code files and Trainium execution logs for papers.

Usage:
    # Download all successful code files
    python download_results.py --code-only
    
    # Download Trainium logs for a specific paper
    python download_results.py --paper-id <paper_id> --logs-only
    
    # Download everything for a specific paper
    python download_results.py --paper-id <paper_id>
    
    # Download all successful code files and logs
    python download_results.py --all
"""

import os
import sys
import boto3
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v2")
CODE_BUCKET = os.getenv("CODE_BUCKET", "papers-code-artifacts")
OUTPUTS_BUCKET = os.getenv("OUTPUTS_BUCKET", "papers-test-outputs")

def get_opensearch_client():
    """Get OpenSearch client"""
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

def get_papers_with_code(os_client) -> List[Dict[str, Any]]:
    """Get all papers that have code generated"""
    query = {
        "query": {
            "term": {"code_generated": True}
        },
        "size": 1000  # Adjust if you have more papers
    }
    
    response = os_client.search(index=OPENSEARCH_INDEX, body=query)
    papers = []
    
    for hit in response['hits']['hits']:
        source = hit['_source']
        papers.append({
            'paper_id': hit['_id'],
            'title': source.get('title', 'Unknown'),
            'code_s3_bucket': source.get('code_s3_bucket', CODE_BUCKET),
            'code_s3_key': source.get('code_s3_key'),
            'code_generated_at': source.get('code_generated_at'),
            'tested': source.get('tested', False),
            'test_success': source.get('test_success', False),
            'stdout_s3_key': source.get('stdout_s3_key'),
            'stderr_s3_key': source.get('stderr_s3_key'),
            'outputs_s3_bucket': source.get('outputs_s3_bucket', OUTPUTS_BUCKET)
        })
    
    return papers

def download_code_file(s3_client, paper: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    """Download a code file from S3"""
    paper_id = paper['paper_id']
    code_key = paper.get('code_s3_key')
    code_bucket = paper.get('code_s3_bucket', CODE_BUCKET)
    
    if not code_key:
        print(f"   ⚠️  No code_s3_key for {paper_id}")
        return None
    
    try:
        # Create filename from paper title
        title = paper.get('title', 'Unknown')
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        safe_title = safe_title.replace(' ', '_')[:80]
        
        filename = f"{paper_id}_{safe_title}.py"
        filepath = output_dir / filename
        
        # Download
        s3_client.download_file(code_bucket, code_key, str(filepath))
        print(f"   ✅ Downloaded: {filename}")
        return filepath
        
    except Exception as e:
        print(f"   ❌ Error downloading code for {paper_id}: {e}")
        return None

def download_trainium_logs(s3_client, paper: Dict[str, Any], output_dir: Path) -> Dict[str, Optional[Path]]:
    """Download Trainium execution logs from S3"""
    paper_id = paper['paper_id']
    outputs_bucket = paper.get('outputs_s3_bucket', OUTPUTS_BUCKET)
    stdout_key = paper.get('stdout_s3_key')
    stderr_key = paper.get('stderr_s3_key')
    
    results = {'stdout': None, 'stderr': None}
    
    # Create logs directory
    logs_dir = output_dir / paper_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Download stdout
    if stdout_key:
        try:
            stdout_path = logs_dir / "stdout.log"
            s3_client.download_file(outputs_bucket, stdout_key, str(stdout_path))
            print(f"   ✅ Downloaded stdout: {stdout_path}")
            results['stdout'] = stdout_path
        except Exception as e:
            print(f"   ⚠️  Error downloading stdout: {e}")
    
    # Download stderr
    if stderr_key:
        try:
            stderr_path = logs_dir / "stderr.log"
            s3_client.download_file(outputs_bucket, stderr_key, str(stderr_path))
            print(f"   ✅ Downloaded stderr: {stderr_path}")
            results['stderr'] = stderr_path
        except Exception as e:
            print(f"   ⚠️  Error downloading stderr: {e}")
    
    # Download metrics if available
    metrics_key = f"{paper_id}/outputs/metrics.json"
    try:
        metrics_path = logs_dir / "metrics.json"
        s3_client.download_file(outputs_bucket, metrics_key, str(metrics_path))
        print(f"   ✅ Downloaded metrics: {metrics_path}")
    except Exception as e:
        # Metrics might not exist, that's okay
        pass
    
    return results

def download_all_code_files(os_client, s3_client, output_dir: Path):
    """Download all successful code files"""
    print("🔍 Finding papers with generated code...")
    papers = get_papers_with_code(os_client)
    
    if not papers:
        print("❌ No papers with generated code found")
        return
    
    print(f"📄 Found {len(papers)} papers with generated code\n")
    
    code_dir = output_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for i, paper in enumerate(papers, 1):
        paper_id = paper['paper_id']
        title = paper.get('title', 'Unknown')[:60]
        print(f"[{i}/{len(papers)}] {paper_id}: {title}...")
        
        if download_code_file(s3_client, paper, code_dir):
            success_count += 1
    
    print(f"\n✅ Downloaded {success_count}/{len(papers)} code files to {code_dir}")

def download_paper_data(os_client, s3_client, paper_id: str, output_dir: Path, code_only: bool = False, logs_only: bool = False):
    """Download data for a specific paper"""
    try:
        doc = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
        source = doc['_source']
        
        # Fetch all fields that might exist in OpenSearch
        # Note: Some fields only exist if execution succeeded (estimated_compute_cost, trainium_hours)
        # Some fields might exist even on failure if Trainium returned partial results
        paper = {
            'paper_id': paper_id,
            'title': source.get('title', 'Unknown'),
            'code_s3_bucket': source.get('code_s3_bucket', CODE_BUCKET),
            'code_s3_key': source.get('code_s3_key'),
            'code_generated': source.get('code_generated', False),
            'tested': source.get('tested', False),
            'test_success': source.get('test_success', False),
            'stdout_s3_key': source.get('stdout_s3_key'),
            'stderr_s3_key': source.get('stderr_s3_key'),
            'outputs_s3_bucket': source.get('outputs_s3_bucket', OUTPUTS_BUCKET),
            # Core execution fields (always present if tested)
            'execution_time': source.get('execution_time'),
            'return_code': source.get('return_code'),
            'timeout': source.get('timeout', False),
            'tested_at': source.get('tested_at'),
            'executed_on': source.get('executed_on'),
            'instance_type': source.get('instance_type'),
            # Performance metrics (may exist even on failure if Trainium returned partial results)
            'peak_memory_mb': source.get('peak_memory_mb'),
            'neuron_core_utilization': source.get('neuron_core_utilization'),
            'throughput_samples_per_sec': source.get('throughput_samples_per_sec'),
            # Training metrics (only if training completed)
            'training_loss': source.get('training_loss'),
            'validation_accuracy': source.get('validation_accuracy'),
            # Cost metrics (only calculated on successful execution)
            'estimated_compute_cost': source.get('estimated_compute_cost'),
            'trainium_hours': source.get('trainium_hours'),
            # Error fields (only present if execution failed)
            'error_message': source.get('error_message'),
            'error_type': source.get('error_type'),
            'has_errors': source.get('has_errors', False),
            # Additional fields that might exist
            'dataset_name': source.get('dataset_name'),
            'lines_of_code': source.get('lines_of_code'),
            'pytorch_version': source.get('pytorch_version')
        }
        
        print(f"📄 Paper: {paper['title']}")
        print(f"   ID: {paper_id}\n")
        
        paper_dir = output_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # Download code
        if not logs_only:
            if paper['code_generated']:
                print("📥 Downloading code file...")
                code_path = download_code_file(s3_client, paper, paper_dir)
                if code_path:
                    print(f"   📁 Saved to: {code_path}\n")
            else:
                print("⚠️  Code not generated for this paper\n")
        
        # Download Trainium logs
        if not code_only:
            if paper['tested']:
                print("📥 Downloading Trainium execution logs...")
                logs = download_trainium_logs(s3_client, paper, paper_dir)
                
                # Show summary with all metrics
                print(f"\n📊 Execution Summary:")
                print(f"   Success: {'✅ Yes' if paper['test_success'] else '❌ No'}")
                if paper.get('tested_at'):
                    print(f"   Tested At: {paper.get('tested_at')}")
                if paper.get('execution_time') is not None:
                    print(f"   Execution Time: {paper.get('execution_time'):.2f}s")
                if paper.get('return_code') is not None:
                    print(f"   Return Code: {paper.get('return_code')}")
                if paper.get('timeout'):
                    print(f"   ⏰ Timeout: Yes")
                
                # Performance Metrics
                if paper.get('peak_memory_mb') is not None:
                    print(f"   💾 Peak Memory: {paper.get('peak_memory_mb'):.1f} MB")
                if paper.get('estimated_compute_cost') is not None:
                    print(f"   💰 Estimated Cost: ${paper.get('estimated_compute_cost'):.2f}")
                if paper.get('trainium_hours') is not None:
                    print(f"   ⏱️  Trainium Hours: {paper.get('trainium_hours'):.4f}")
                if paper.get('neuron_core_utilization') is not None:
                    print(f"   🔧 Neuron Core Utilization: {paper.get('neuron_core_utilization')}")
                if paper.get('throughput_samples_per_sec') is not None:
                    print(f"   📈 Throughput: {paper.get('throughput_samples_per_sec'):.2f} samples/sec")
                
                # Training Metrics
                if paper.get('training_loss') is not None:
                    print(f"   📉 Training Loss: {paper.get('training_loss')}")
                if paper.get('validation_accuracy') is not None:
                    print(f"   📊 Validation Accuracy: {paper.get('validation_accuracy')}")
                
                # Error info if failed
                if not paper.get('test_success') and paper.get('error_message'):
                    print(f"   ⚠️  Error: {paper.get('error_message')}")
                    if paper.get('error_type'):
                        print(f"   🔍 Error Type: {paper.get('error_type')}")
                
                # Show full stdout if available
                if logs.get('stdout'):
                    print(f"\n📋 Full STDOUT Log:")
                    print("=" * 80)
                    with open(logs['stdout'], 'r') as f:
                        print(f.read())
                    print("=" * 80)
                
                # Show full stderr if available
                if logs.get('stderr'):
                    print(f"\n📋 Full STDERR Log:")
                    print("=" * 80)
                    with open(logs['stderr'], 'r') as f:
                        stderr_content = f.read()
                        if stderr_content.strip():
                            print(stderr_content)
                        else:
                            print("(empty)")
                    print("=" * 80)
            else:
                print("⚠️  Paper not tested on Trainium yet\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Download code files and Trainium execution logs')
    parser.add_argument('--paper-id', help='Specific paper ID to download')
    parser.add_argument('--code-only', action='store_true', help='Only download code files')
    parser.add_argument('--logs-only', action='store_true', help='Only download Trainium logs')
    parser.add_argument('--all', action='store_true', help='Download all successful code files')
    parser.add_argument('--output-dir', default='downloaded_results', help='Output directory (default: downloaded_results)')
    
    args = parser.parse_args()
    
    # Initialize clients
    os_client = get_opensearch_client()
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        # Download all code files
        download_all_code_files(os_client, s3_client, output_dir)
    elif args.paper_id:
        # Download specific paper
        download_paper_data(os_client, s3_client, args.paper_id, output_dir, 
                          code_only=args.code_only, logs_only=args.logs_only)
    else:
        parser.print_help()
        print("\n💡 Examples:")
        print("   # Download all successful code files:")
        print("   python download_results.py --all")
        print("\n   # Download everything for a specific paper:")
        print("   python download_results.py --paper-id <paper_id>")
        print("\n   # Download only Trainium logs for a paper:")
        print("   python download_results.py --paper-id <paper_id> --logs-only")

if __name__ == "__main__":
    main()

