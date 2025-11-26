import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# DynamoDB Table for storing execution errors.
ERROR_DB_TABLE_NAME = os.getenv('ERROR_DB_TABLE_NAME')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Initialize DynamoDB client
dynamodb_client = None
dynamodb_resource = None

if ERROR_DB_TABLE_NAME:
    try:
        dynamodb_client = boto3.client('dynamodb', region_name=AWS_REGION)
        dynamodb_resource = boto3.resource('dynamodb', region_name=AWS_REGION)
        logger.info(f"DynamoDB client initialized for table: {ERROR_DB_TABLE_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize DynamoDB client: {e}")
else:
    logger.warning("ERROR_DB_TABLE_NAME not set.")


def _get_partition_key(paper_id: str) -> str:
    return f"DOC#{paper_id}"


def _get_sort_key(iteration: int, error_id: str) -> str:
    return f"ITER#{iteration}#ERR#{error_id}"


def _parse_sort_key(sort_key: str) -> tuple:
    """Parse sort key to extract iteration and error_id"""
    # Format: ITER#<iteration>#ERR#<error_id>
    try:
        parts = sort_key.split('#')
        if len(parts) >= 4 and parts[0] == 'ITER' and parts[2] == 'ERR':
            iteration = int(parts[1])
            error_id = '#'.join(parts[3:])  # In case error_id contains #
            return iteration, error_id
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse sort key {sort_key}: {e}")
    return None, None


def save_error(paper_id: str, error_data: Dict[str, Any]) -> str:
    """
    Save error to DynamoDB.
    
    Args:
        paper_id: Paper/document ID
        error_data: Error information dictionary
        
    Returns:
        DynamoDB item identifier (partition_key#sort_key)
    """
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return ""
    
    try:
        # Get current error count to determine iteration
        current_count = get_error_count(paper_id)
        iteration = current_count + 1
        
        # Generate unique error ID
        error_id = uuid.uuid4().hex
        
        # Generate keys
        partition_key = _get_partition_key(paper_id)
        sort_key = _get_sort_key(iteration, error_id)
        
        # Prepare item
        timestamp = datetime.now().isoformat()
        item = {
            'partition_key': partition_key,
            'sort_key': sort_key,
            'paper_id': paper_id,
            'iteration': iteration,
            'error_id': error_id,
            'timestamp': timestamp,
            'error_data': json.dumps(error_data) if isinstance(error_data, dict) else str(error_data),
            # Store individual error fields for easier querying
            'stderr': error_data.get('stderr', ''),
            'stdout': error_data.get('stdout', ''),
            'error_message': error_data.get('error_message', ''),
            'error_type': error_data.get('error_type', 'execution_error'),
            'return_code': error_data.get('return_code', -1),
            'execution_time': error_data.get('execution_time', 0)
        }
        
        # Write to DynamoDB
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        table.put_item(Item=item)
        
        logger.info(f"Saved error for {paper_id} (iteration {iteration}, total errors: {iteration})")
        return f"{partition_key}#{sort_key}"
        
    except ClientError as e:
        logger.error(f"DynamoDB error saving error for {paper_id}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error saving error for {paper_id}: {e}")
        return ""


def get_errors(paper_id: str) -> List[Dict[str, Any]]:
    
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return []
    
    try:
        partition_key = _get_partition_key(paper_id)
        
        # Query DynamoDB by partition key
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        response = table.query(
            KeyConditionExpression='partition_key = :pk',
            ExpressionAttributeValues={
                ':pk': partition_key
            },
            ScanIndexForward=True  # Sort ascending by sort key (iteration order)
        )
        
        errors = []
        for item in response.get('Items', []):
            # Parse error_data
            error_data_str = item.get('error_data', '{}')
            try:
                error_data = json.loads(error_data_str) if isinstance(error_data_str, str) else error_data_str
            except json.JSONDecodeError:
                # Fallback: reconstruct from individual fields
                error_data = {
                    'stderr': item.get('stderr', ''),
                    'stdout': item.get('stdout', ''),
                    'error_message': item.get('error_message', ''),
                    'error_type': item.get('error_type', 'execution_error'),
                    'return_code': item.get('return_code', -1),
                    'execution_time': item.get('execution_time', 0)
                }
            
            errors.append({
                'timestamp': item.get('timestamp', ''),
                'error_data': error_data
            })
        
        return errors
        
    except ClientError as e:
        logger.error(f"DynamoDB error getting errors for {paper_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting errors for {paper_id}: {e}")
        return []



def clear_errors(paper_id: str) -> bool:
    
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return False
    
    try:
        partition_key = _get_partition_key(paper_id)
        
        # Query to get all items
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        response = table.query(
            KeyConditionExpression='partition_key = :pk',
            ExpressionAttributeValues={
                ':pk': partition_key
            }
        )
        
        # Delete all items
        items = response.get('Items', [])
        if not items:
            logger.info(f"No errors to clear for {paper_id}")
            return True
        
        # Batch delete
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={
                        'partition_key': item['partition_key'],
                        'sort_key': item['sort_key']
                    }
                )
        
        logger.info(f"Cleared {len(items)} errors for {paper_id}")
        return True
        
    except ClientError as e:
        logger.error(f"DynamoDB error clearing errors for {paper_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error clearing errors for {paper_id}: {e}")
        return False