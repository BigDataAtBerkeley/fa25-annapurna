#!/usr/bin/env python3
"""
Download generated code from S3 to local generated_code folder.
Usage: python download_generated_code.py <paper_id>
"""

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import os
import sys
import time
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configuration
region = os.getenv("AWS_REGION", "us-east-1")
host = os.getenv("OPENSEARCH_ENDPOINT")
index_name = os.getenv("OPENSEARCH_INDEX", "research-papers-v2")
code_bucket = os.getenv("CODE_BUCKET", "papers-code-artifacts")

if not host:
    print("Error: OPENSEARCH_ENDPOINT environment variable not set")
    sys.exit(1)

# Get paper ID from command line
if len(sys.argv) < 2:
    print("Usage: python download_generated_code.py <paper_id>")
    sys.exit(1)

paper_id = sys.argv[1]

# Auth setup
session = boto3.Session(region_name=region)
credentials = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(credentials, region, "es")

# OpenSearch client
os_client = OpenSearch(
    hosts=[{"host": host.replace("https://", "").replace("http://", ""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)

# S3 client
s3_client = boto3.client('s3', region_name=region)

# Wait for code generation (poll OpenSearch)
print(f"Waiting for code generation for paper: {paper_id}")
max_wait = 300  # 5 minutes
wait_interval = 10  # Check every 10 seconds
elapsed = 0

while elapsed < max_wait:
    try:
        doc = os_client.get(index=index_name, id=paper_id)
        source = doc['_source']
        
        if source.get('code_generated') and source.get('code_s3_key'):
            print(f"✅ Code generation complete!")
            print(f"   Paper: {source.get('title', 'Unknown')}")
            print(f"   S3 Bucket: {source.get('code_s3_bucket')}")
            print(f"   S3 Key: {source.get('code_s3_key')}")
            break
        else:
            print(f"   Still generating... ({elapsed}s elapsed)")
            time.sleep(wait_interval)
            elapsed += wait_interval
    except Exception as e:
        print(f"   Error checking status: {e}")
        time.sleep(wait_interval)
        elapsed += wait_interval

if elapsed >= max_wait:
    print(f"❌ Timeout waiting for code generation")
    sys.exit(1)

# Download code from S3
try:
    s3_bucket = source.get('code_s3_bucket', code_bucket)
    code_key = source.get('code_s3_key')
    metadata_key = source.get('code_metadata_s3_key')
    
    if not code_key:
        print("❌ No code_s3_key found in OpenSearch")
        sys.exit(1)
    
    # Create generated_code directory if it doesn't exist
    output_dir = "generated_code"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename from paper title
    paper_title = source.get('title', 'Unknown_Paper')
    # Sanitize filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in paper_title)
    safe_title = safe_title.replace(' ', '_')[:100]  # Limit length
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    code_filename = f"{safe_title}_{timestamp}.py"
    code_path = os.path.join(output_dir, code_filename)
    
    # Download code
    print(f"\n📥 Downloading code from S3...")
    s3_client.download_file(s3_bucket, code_key, code_path)
    print(f"✅ Code saved to: {code_path}")
    
    # Download metadata if available
    if metadata_key:
        metadata_filename = f"{safe_title}_{timestamp}_metadata.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        s3_client.download_file(s3_bucket, metadata_key, metadata_path)
        print(f"✅ Metadata saved to: {metadata_path}")
    
    print(f"\n✅ Successfully downloaded generated code!")
    
except Exception as e:
    print(f"❌ Error downloading code: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

