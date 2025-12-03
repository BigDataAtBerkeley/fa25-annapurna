import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# DynamoDB Table for storing execution errors.
ERROR_DB_TABLE_NAME = os.getenv('ERROR_DB_TABLE_NAME', 'docRunErrors')
# Use DYNAMODB_REGION if set, otherwise fall back to AWS_REGION, default to us-east-2
DYNAMODB_REGION = os.getenv('DYNAMODB_REGION') or os.getenv('AWS_REGION', 'us-east-2')
AWS_REGION = DYNAMODB_REGION  # For backward compatibility

# Initialize DynamoDB client
dynamodb_client = None
dynamodb_resource = None

if ERROR_DB_TABLE_NAME:
    try:
        dynamodb_client = boto3.client('dynamodb', region_name=DYNAMODB_REGION)
        dynamodb_resource = boto3.resource('dynamodb', region_name=DYNAMODB_REGION)
        logger.info(f"DynamoDB client initialized for table: {ERROR_DB_TABLE_NAME} in region: {DYNAMODB_REGION}")
    except Exception as e:
        logger.error(f"Failed to initialize DynamoDB client: {e}")
else:
    logger.warning("ERROR_DB_TABLE_NAME not set.")


def _get_partition_key(paper_id: str) -> str:
    """
    Return partition key for DynamoDB.
    Uses DOC# prefix for backward compatibility with existing data.
    """
    if paper_id.startswith('DOC#'):
        return paper_id  # Already has prefix
    return f'DOC#{paper_id}'  # Add prefix for backward compatibility


def _get_sort_key(iteration: int, error_id: str = None) -> str:
    """
    Return sort key for DynamoDB.
    Uses ITER#<iteration>#ERR#<error_id> format for backward compatibility.
    If error_id is not provided, generates one.
    """
    import uuid
    if error_id is None:
        error_id = str(uuid.uuid4()).replace('-', '')
    return f'ITER#{iteration}#ERR#{error_id}'


def _parse_sort_key(sort_key: str) -> tuple:
    """Parse sort key to extract iteration.
    
    New format: just the iteration number as a string (e.g., "1", "2")
    Old format (for backward compatibility): ITER#<iteration>#ERR#<error_id>
    """
    try:
        # Try new format first: just a number as string
        iteration = int(sort_key)
        return iteration, None
    except ValueError:
        # Try old format for backward compatibility
        try:
            parts = sort_key.split('#')
            if len(parts) >= 4 and parts[0] == 'ITER' and parts[2] == 'ERR':
                iteration = int(parts[1])
                error_id = '#'.join(parts[3:])  # In case error_id contains #
                return iteration, error_id
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse sort key {sort_key}: {e}")
    return None, None

def get_error_count(paper_id: str) -> int:
    """
    Get the count of errors for a given paper.
    
    Args:
        paper_id: Paper/document ID
        
    Returns:
        Number of errors stored for this paper
    """
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return 0
    
    try:
        partition_key = _get_partition_key(paper_id)
        
        # Query DynamoDB with Select='COUNT' for efficiency
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        response = table.query(
            KeyConditionExpression='docID = :pk',
            ExpressionAttributeValues={
                ':pk': partition_key
            },
            Select='COUNT'
        )
        
        count = response.get('Count', 0)
        logger.debug(f"Error count for {paper_id}: {count}")
        return count
        
    except ClientError as e:
        logger.error(f"DynamoDB error getting error count for {paper_id}: {e}")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error getting error count for {paper_id}: {e}")
        return 0

def save_error(paper_id: str, error_data: Dict[str, Any], iteration: Optional[int] = None) -> str:
    """
    Save error to DynamoDB.
    
    Args:
        paper_id: Paper/document ID
        error_data: Error information dictionary
        iteration: Optional iteration number (if not provided, will be calculated automatically)
        
    Returns:
        DynamoDB item identifier (partition_key#sort_key)
    """
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return ""
    
    try:
        # Get iteration number (use provided or calculate)
        if iteration is None:
            # Get current error count to determine iteration
            current_count = get_error_count(paper_id)
            iteration = current_count + 1
        else:
            # Use provided iteration number
            iteration = int(iteration)
        
        # Generate unique error ID
        error_id = uuid.uuid4().hex
        
        # Generate keys (using old format with prefixes for backward compatibility)
        partition_key = _get_partition_key(paper_id)
        sort_key = _get_sort_key(iteration, error_id)
        
        # Prepare item
        timestamp = datetime.now().isoformat()
        item = {
            'docID': partition_key,  # Use 'docID' to match table schema
            'interationNum': sort_key,
            'paper_id': paper_id,
            'iteration': iteration,
            'error_id': error_id,
            'timestamp': timestamp,
            'error_data': json.dumps(error_data) if isinstance(error_data, dict) else str(error_data),
            'stderr': error_data.get('stderr', ''),
            'stdout': error_data.get('stdout', ''),
            'error_message': error_data.get('error_message', ''),
            'error_type': error_data.get('error_type', 'execution_error'),
            'return_code': int(error_data.get('return_code', -1)),
            'execution_time': Decimal(str(error_data.get('execution_time', 0)))
        }
        
        # Add fixes_applied if present in error_data
        if 'fixes_applied' in error_data:
            item['fixes_applied'] = json.dumps(error_data['fixes_applied']) if isinstance(error_data['fixes_applied'], (dict, list)) else str(error_data['fixes_applied'])
        
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
        
        # Query DynamoDB by partition key with pagination support
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        errors = []
        
        # Handle pagination to get ALL errors
        query_kwargs = {
            'KeyConditionExpression': 'docID = :pk',
            'ExpressionAttributeValues': {
                ':pk': partition_key
            },
            'ScanIndexForward': True  # Sort ascending by sort key (iteration order)
        }
        
        while True:
            response = table.query(**query_kwargs)
            
            for item in response.get('Items', []):
                # Parse error_data
                error_data_str = item.get('error_data', '{}')
                try:
                    error_data = json.loads(error_data_str) if isinstance(error_data_str, str) else error_data_str
                except json.JSONDecodeError:
                    # Fallback: reconstruct from individual fields
                    # Convert Decimal back to float for execution_time
                    execution_time = item.get('execution_time', 0)
                    error_data = {
                        'stderr': item.get('stderr', ''),
                        'stdout': item.get('stdout', ''),
                        'error_message': item.get('error_message', ''),
                        'error_type': item.get('error_type', 'execution_error'),
                        'return_code': int(item.get('return_code', -1)),
                        'execution_time': float(execution_time) if isinstance(execution_time, Decimal) else float(execution_time or 0)
                    }
                
                # Extract iteration from item or parse from sort key
                iteration = item.get('iteration')
                if iteration is None:
                    # Try to parse from sort key
                    sort_key = item.get('interationNum', '')
                    parsed_iteration, _ = _parse_sort_key(sort_key)
                    iteration = parsed_iteration
                
                error_item = {
                    'timestamp': item.get('timestamp', ''),
                    'iteration': iteration,
                    'error_data': error_data
                }
                
                # Parse fixes_applied if present
                fixes_applied_str = item.get('fixes_applied')
                if fixes_applied_str:
                    try:
                        fixes_applied = json.loads(fixes_applied_str) if isinstance(fixes_applied_str, str) else fixes_applied_str
                        error_item['fixes_applied'] = fixes_applied
                    except json.JSONDecodeError:
                        pass
                
                errors.append(error_item)
            
            # Check if there are more items (pagination)
            if 'LastEvaluatedKey' not in response:
                break
            query_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
        return errors
        
    except ClientError as e:
        logger.error(f"DynamoDB error getting errors for {paper_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting errors for {paper_id}: {e}")
        return []

def get_errors_for_paper_ids(paper_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get errors from DynamoDB for specific paper IDs.
    
    Args:
        paper_ids: List of paper IDs to get errors for
        
    Returns:
        List of error dictionaries with paper_id included
    """
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.warning("DynamoDB table name not configured")
        return []
    
    all_errors = []
    try:
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        
        for paper_id in paper_ids:
            try:
                partition_key = _get_partition_key(paper_id)
                
                response = table.query(
                    KeyConditionExpression='docID = :pk',
                    ExpressionAttributeValues={':pk': partition_key},
                    ScanIndexForward=True
                )
                
                for item in response.get('Items', []):
                    error_data_str = item.get('error_data', '{}')
                    try:
                        error_data = json.loads(error_data_str) if isinstance(error_data_str, str) else error_data_str
                    except json.JSONDecodeError:
                        execution_time = item.get('execution_time', 0)
                        error_data = {
                            'stderr': item.get('stderr', ''),
                            'stdout': item.get('stdout', ''),
                            'error_message': item.get('error_message', ''),
                            'error_type': item.get('error_type', 'execution_error'),
                            'return_code': int(item.get('return_code', -1)),
                            'execution_time': float(execution_time) if isinstance(execution_time, Decimal) else float(execution_time or 0)
                        }
                    
                    all_errors.append({
                        'paper_id': paper_id,
                        'timestamp': item.get('timestamp', ''),
                        'error_data': error_data
                    })
                    
            except Exception as e:
                logger.warning(f"Error retrieving errors for paper {paper_id}: {e}")
                continue
        
        return all_errors
        
    except Exception as e:
        logger.error(f"Error retrieving errors from DynamoDB: {e}")
        return []


def update_error_fixes(paper_id: str, fixes_applied: Dict[str, Any]) -> bool:
    """
    Update the most recent error record with fixes_applied information.
    
    Args:
        paper_id: Paper/document ID
        fixes_applied: Dictionary containing fix information
        
    Returns:
        True if update successful, False otherwise
    """
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return False
    
    try:
        partition_key = _get_partition_key(paper_id)
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        
        # Get the most recent error
        response = table.query(
            KeyConditionExpression='docID = :pk',
            ExpressionAttributeValues={':pk': partition_key},
            ScanIndexForward=False,  # Get most recent first
            Limit=1
        )
        
        items = response.get('Items', [])
        if not items:
            logger.warning(f"No errors found for {paper_id} to update with fixes")
            return False
        
        item = items[0]
        sort_key = item.get('interationNum')
        
        # Update the item with fixes_applied
        table.update_item(
            Key={
                'docID': partition_key,
                'interationNum': sort_key
            },
            UpdateExpression='SET fixes_applied = :fixes',
            ExpressionAttributeValues={
                ':fixes': json.dumps(fixes_applied) if isinstance(fixes_applied, (dict, list)) else str(fixes_applied)
            }
        )
        
        logger.info(f"Updated most recent error for {paper_id} with fixes_applied")
        return True
        
    except Exception as e:
        logger.error(f"Error updating error fixes for {paper_id}: {e}")
        return False


def clear_errors(paper_id: str) -> bool:
    
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return False
    
    try:
        partition_key = _get_partition_key(paper_id)
        
        # Query to get all items
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        response = table.query(
            KeyConditionExpression='docID = :pk',
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
                        'docID': item['docID'],  
                        'interationNum': item['interationNum']
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


def get_all_errors(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Get errors from all papers (for proactive error checking in code review).
    
    Args:
        limit: Maximum number of errors to return
        
    Returns:
        List of error dictionaries from all papers
    """
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.warning("DynamoDB table name not configured - cannot retrieve all errors")
        return []
    
    try:
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        
        # Scan table to get errors from all papers
        errors = []
        scan_kwargs = {
            'Limit': limit
        }
        
        while len(errors) < limit:
            response = table.scan(**scan_kwargs)
            items = response.get('Items', [])
            
            for item in items:
                if len(errors) >= limit:
                    break
                    
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
                        'return_code': int(item.get('return_code', -1)),
                        'execution_time': float(item.get('execution_time', 0))
                    }
                
                # Extract paper_id from partition key
                # New format: just paper_id, Old format: DOC#<paper_id> (for backward compatibility)
                doc_id = item.get('docID', '')
                if doc_id.startswith('DOC#'):
                    paper_id = doc_id.replace('DOC#', '')  # Old format
                else:
                    paper_id = doc_id  # New format (just paper_id)
                
                errors.append({
                    'paper_id': paper_id,
                    'timestamp': item.get('timestamp', ''),
                    'error_data': error_data
                })
            
            # Check if there are more items
            if 'LastEvaluatedKey' not in response:
                break
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
        logger.info(f"Retrieved {len(errors)} errors from all papers (limit: {limit})")
        return errors[:limit]
        
    except ClientError as e:
        logger.error(f"DynamoDB error getting all errors: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting all errors: {e}")
        return []