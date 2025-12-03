"""
S3 Code Storage Module

Manages code storage in S3 bucket papers-code-artifacts.
Each paper_id has one code file that gets replaced on updates.
"""

import os
import json
import logging
import boto3
from typing import Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = os.getenv('CODE_STORAGE_BUCKET', 'papers-code-artifacts')
S3_PREFIX = 'code'  # Files stored as: code/{paper_id}.py

s3_client = None
try:
    s3_client = boto3.client('s3')
    logger.info(f"S3 code storage initialized: bucket={S3_BUCKET}, prefix={S3_PREFIX}")
except Exception as e:
    logger.warning(f"Failed to initialize S3 client: {e}")


def save_code(paper_id: str, code: str) -> str:
    """
    Save code to S3, replacing any existing code for this paper_id.
    
    Args:
        paper_id: Paper/document ID
        code: Python code to save
        
    Returns:
        S3 key of saved file
    """
    if not s3_client:
        logger.error("S3 client not initialized")
        return None
    
    s3_key = f"{S3_PREFIX}/{paper_id}.py"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=code.encode('utf-8'),
            ContentType='text/x-python'
        )
        s3_path = f"s3://{S3_BUCKET}/{s3_key}"
        logger.info(f"Saved code to S3: {s3_path}")
        return s3_key
    except ClientError as e:
        logger.error(f"Failed to save code to S3: {e}")
        return None


def get_code(paper_id: str) -> Optional[str]:
    """
    Get code from S3 for a paper_id.
    
    Args:
        paper_id: Paper/document ID
        
    Returns:
        Code string or None if not found
    """
    if not s3_client:
        logger.error("S3 client not initialized")
        return None
    
    s3_key = f"{S3_PREFIX}/{paper_id}.py"
    
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )
        code = response['Body'].read().decode('utf-8')
        logger.info(f"Retrieved code from S3: {s3_key}")
        return code
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logger.info(f"No code found in S3 for {paper_id}")
            return None
        logger.error(f"Failed to get code from S3: {e}")
        return None


def code_exists(paper_id: str) -> bool:
    """Check if code exists in S3 for a paper_id."""
    if not s3_client:
        return False
    
    s3_key = f"{S3_PREFIX}/{paper_id}.py"
    
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logger.error(f"Failed to check code existence: {e}")
        return False

