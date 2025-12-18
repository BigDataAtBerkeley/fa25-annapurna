#!/usr/bin/env python3
"""
Check OpenSearch index mapping (schema) to see all defined fields
"""

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v3")

if not OPENSEARCH_ENDPOINT:
    raise EnvironmentError("OPENSEARCH_ENDPOINT environment variable is required")

# Setup client
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")

os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://","").replace("http://",""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60
)

def check_mapping():
    """Display the index mapping (schema) showing all defined fields"""
    try:
        # Get mapping
        mapping = os_client.indices.get_mapping(index=OPENSEARCH_INDEX)
        
        print("="*80)
        print(f"OpenSearch Index Mapping for: {OPENSEARCH_INDEX}")
        print("="*80)
        
        # Extract properties (fields)
        properties = mapping[OPENSEARCH_INDEX]['mappings'].get('properties', {})
        
        print(f"\nTotal fields defined: {len(properties)}\n")
        
        # Display fields sorted alphabetically
        for field_name in sorted(properties.keys()):
            field_config = properties[field_name]
            field_type = field_config.get('type', 'nested/object')
            
            # Handle nested fields
            if 'properties' in field_config:
                print(f"📦 {field_name} (nested object):")
                for nested_field, nested_config in field_config['properties'].items():
                    nested_type = nested_config.get('type', 'unknown')
                    print(f"    └─ {nested_field}: {nested_type}")
            else:
                print(f"📄 {field_name}: {field_type}")
        
        print("\n" + "="*80)
        print("Full mapping JSON:")
        print("="*80)
        print(json.dumps(mapping, indent=2))
        
    except Exception as e:
        print(f"❌ Error getting mapping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_mapping()

