"""
Lambda handler for PapersCodeGenerator.

This is the entry point for the Lambda function.
Uses OpenSearch for storage and SQS for queueing.
CALLS main_handler.py 
"""

import json
import os
import boto3
import logging
from datetime import datetime
from typing import Dict, Any
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# Import existing code gen handler
import sys
sys.path.append('/opt/python')  # Lambda layer path
sys.path.append(os.path.dirname(__file__))

from code_gen.main_handler import CodeGenHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# AWS clients
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')
session = boto3.Session()

# Environment variables
CODE_BUCKET = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
CODE_QUEUE_URL = os.getenv('CODE_QUEUE_URL')
OPENSEARCH_ENDPOINT = os.getenv('OPENSEARCH_ENDPOINT')
OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'research-papers-v2')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Setup OpenSearch client
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

def save_to_s3(paper_id: str, code: str, metadata: Dict[str, Any]) -> Dict[str, str]:
    """Save generated code and metadata to S3."""
    prefix = f"{paper_id}/"
    
    # Save code
    code_key = f"{prefix}code.py"
    s3_client.put_object(
        Bucket=CODE_BUCKET,
        Key=code_key,
        Body=code.encode('utf-8'),
        ContentType='text/x-python'
    )
    
    # Save metadata
    metadata_key = f"{prefix}metadata.json"
    s3_client.put_object(
        Bucket=CODE_BUCKET,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    
    logger.info(f"Saved code to s3://{CODE_BUCKET}/{code_key}")
    
    return {
        "s3_bucket": CODE_BUCKET,
        "code_key": code_key,
        "metadata_key": metadata_key
    }

def update_opensearch(paper_id: str, code_info: Dict[str, Any]):
    """Update OpenSearch document with code generation status."""
    try:
        os_client.update(
            index=OPENSEARCH_INDEX,
            id=paper_id,
            body={
                "doc": {
                    "code_generated": True,
                    "code_s3_bucket": code_info["s3_bucket"],
                    "code_s3_key": code_info["code_key"],
                    "code_generated_at": datetime.now().isoformat(),
                    "code_metadata_s3_key": code_info["metadata_key"]
                }
            }
        )
        logger.info(f"Updated OpenSearch document {paper_id} with code generation status")
        
    except Exception as e:
        logger.error(f"Error updating OpenSearch: {e}")
        raise

def send_to_sqs(paper_id: str, paper_title: str, s3_info: Dict[str, str]):
    """Send message to SQS for code testing."""
    if not CODE_QUEUE_URL:
        logger.warning("CODE_QUEUE_URL not set, skipping SQS")
        return
    
    message = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "s3_bucket": s3_info["s3_bucket"],
        "s3_code_key": s3_info["code_key"],
        "s3_metadata_key": s3_info["metadata_key"],
        "queued_at": datetime.now().isoformat()
    }
    
    sqs_client.send_message(
        QueueUrl=CODE_QUEUE_URL,
        MessageBody=json.dumps(message),
        MessageGroupId=paper_id,
        MessageDeduplicationId=f"{paper_id}-{datetime.now().isoformat()}"
    )
    
    logger.info(f"Sent message to SQS for paper {paper_id}")

def lambda_handler(event, context):
    """Lambda handler that generates code, saves to S3, updates OpenSearch, and queues for testing."""
    try:
        logger.info("PapersCodeGenerator invoked")
        
        # Generate code using existing handler
        handler = CodeGenHandler()
        result = handler.handle_lambda_event(event)
        
        if not result.get('success'):
            return result
        
        # Extract result (handle both single and multiple results)
        if 'results' in result:
            code_result = result['results'][0]
        else:
            code_result = result
        
        if not code_result.get('success'):
            return code_result
        
        # Extract details
        paper_id = code_result.get('paper_id')
        paper_title = code_result.get('paper_title')
        code = code_result.get('code')
        
        # Save to S3
        s3_info = save_to_s3(paper_id, code, code_result)
        
        # Update OpenSearch
        update_opensearch(paper_id, s3_info)
        
        # Send to SQS for testing
        send_to_sqs(paper_id, paper_title, s3_info)
        
        # Add info to response
        code_result['s3_bucket'] = s3_info['s3_bucket']
        code_result['s3_code_key'] = s3_info['code_key']
        code_result['queued_for_testing'] = True
        code_result['opensearch_updated'] = True
        
        return code_result
        
    except Exception as e:
        logger.error(f"Error in PapersCodeGenerator: {e}")
        return {
            "success": False,
            "error": str(e)
        }

