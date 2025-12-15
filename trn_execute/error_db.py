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

ERROR_DB_TABLE_NAME = os.getenv('ERROR_DB_TABLE_NAME', 'docRunErrors')
DYNAMODB_REGION = os.getenv('DYNAMODB_REGION', 'us-east-2')

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
    if paper_id.startswith('DOC#'):
        return paper_id  
    return f'DOC#{paper_id}'  


def _get_sort_key(iteration: int, error_id: str = None) -> str:
    import uuid
    if error_id is None:
        error_id = str(uuid.uuid4()).replace('-', '')
    return f'ITER#{iteration}#ERR#{error_id}'


def _parse_sort_key(sort_key: str) -> tuple:
    try:
        iteration = int(sort_key)
        return iteration, None
    except ValueError:
        try:
            parts = sort_key.split('#')
            if len(parts) >= 4 and parts[0] == 'ITER' and parts[2] == 'ERR':
                iteration = int(parts[1])
                error_id = '#'.join(parts[3:]) 
                return iteration, error_id
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse sort key {sort_key}: {e}")
    return None, None

def get_error_count(paper_id: str) -> int:
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return 0
    
    try:
        partition_key = _get_partition_key(paper_id)
        
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
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.error("DynamoDB table name not configured")
        return ""
    
    try:
        if iteration is None:
            current_count = get_error_count(paper_id)
            iteration = current_count + 1
        else:
            iteration = int(iteration)
        
        error_id = uuid.uuid4().hex
        
        partition_key = _get_partition_key(paper_id)
        sort_key = _get_sort_key(iteration, error_id)
        
        # Prepare information for the error
        timestamp = datetime.now().isoformat()
        item = {
            'docID': partition_key,
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
        
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        errors = []
        
        query_kwargs = {
            'KeyConditionExpression': 'docID = :pk',
            'ExpressionAttributeValues': {
                ':pk': partition_key
            },
            'ScanIndexForward': True  
        }
        
        while True:
            response = table.query(**query_kwargs)
            
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
                
                # Extract iteration from item or parse from sort key
                iteration = item.get('iteration')
                if iteration is None:
                    sort_key = item.get('interationNum', '')
                    parsed_iteration, _ = _parse_sort_key(sort_key)
                    iteration = parsed_iteration
                
                error_item = {
                    'timestamp': item.get('timestamp', ''),
                    'iteration': iteration,
                    'error_data': error_data
                }
                
                fixes_applied_str = item.get('fixes_applied')
                if fixes_applied_str:
                    try:
                        fixes_applied = json.loads(fixes_applied_str) if isinstance(fixes_applied_str, str) else fixes_applied_str
                        error_item['fixes_applied'] = fixes_applied
                    except json.JSONDecodeError:
                        pass
                
                errors.append(error_item)
            
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
    # Updates the most recent error with the fixes applied
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
    if not ERROR_DB_TABLE_NAME or not dynamodb_client:
        logger.warning("DynamoDB table name not configured - cannot retrieve all errors")
        return []
    
    try:
        table = dynamodb_resource.Table(ERROR_DB_TABLE_NAME)
        
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
                
                doc_id = item.get('docID', '')
                if doc_id.startswith('DOC#'):
                    paper_id = doc_id.replace('DOC#', '')  
                else:
                    paper_id = doc_id  
                
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