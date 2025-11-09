#!/usr/bin/env python3
"""
Script to reset code_generated flag in OpenSearch for testing new models.
Usage: python reset_code_generated.py <paper_id>
"""

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Configuration
region = os.getenv("AWS_REGION", "us-east-1")
host = os.getenv("OPENSEARCH_ENDPOINT")
index_name = os.getenv("OPENSEARCH_INDEX", "research-papers-v2")

if not host:
    print("Error: OPENSEARCH_ENDPOINT environment variable not set")
    sys.exit(1)

# Get paper ID from command line
if len(sys.argv) < 2:
    print("Usage: python reset_code_generated.py <paper_id>")
    print("\nExample:")
    print("  python reset_code_generated.py c03b86be12ac30b75fbeeda5c74a2dec46e335af4ae2ce84a86566055338e122")
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
    timeout=60  # Increase timeout
)

try:
    # First, get the current document to see what we're updating
    print(f"Fetching paper: {paper_id}")
    current_doc = os_client.get(index=index_name, id=paper_id)
    paper_title = current_doc['_source'].get('title', 'Unknown')
    print(f"Paper: {paper_title}")
    print(f"Current code_generated: {current_doc['_source'].get('code_generated', 'Not set')}")
    
    # Update the document
    print(f"\nResetting code_generated to False and clearing S3 references...")
    
    # Use script to remove fields (OpenSearch doesn't support None/null)
    update_body = {
        "script": {
            "source": """
                ctx._source.code_generated = false;
                ctx._source.remove('code_s3_bucket');
                ctx._source.remove('code_s3_key');
                ctx._source.remove('code_metadata_s3_key');
            """,
            "lang": "painless"
        }
    }
    
    result = os_client.update(
        index=index_name,
        id=paper_id,
        body=update_body,
        refresh=True
    )
    
    print(f"✅ Successfully updated paper {paper_id}")
    print(f"   Result: {result.get('result')}")
    print(f"\nYou can now regenerate code for this paper using:")
    print(f"  aws lambda invoke --function-name PapersCodeGenerator \\")
    print(f"    --payload '{{\"action\":\"generate_by_id\",\"paper_id\":\"{paper_id}\"}}' \\")
    print(f"    --cli-binary-format raw-in-base64-out response.json")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

