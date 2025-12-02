"""
AWS DynamoDB client for retrieving execution errors from the docRunErrors table.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DynamoClient:
    """Client for interacting with AWS DynamoDB error database."""
    
    def __init__(self, region_name: Optional[str] = None, table_name: Optional[str] = None):
        """
        Initialize DynamoDB client for error database.
        
        Args:
            region_name: AWS region for DynamoDB (default: us-east-2)
            table_name: DynamoDB table name (default: docRunErrors)
        """
        self.aws_region = region_name or os.getenv('ERROR_DB_REGION', 'us-east-2')
        self.table_name = table_name or os.getenv('ERROR_DB_TABLE_NAME', 'docRunErrors')
        
        try:
            self.client = boto3.client('dynamodb', region_name=self.aws_region)
            self.resource = boto3.resource('dynamodb', region_name=self.aws_region)
            logger.info(f"DynamoDB client initialized (region: {self.aws_region}, table: {self.table_name})")
        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB client: {e}")
            self.client = None
            self.resource = None
    
    def _get_partition_key(self, paper_id: str) -> str:
        """
        Build partition key for DynamoDB query.
        Format: DOC#<paper_id>
        
        Args:
            paper_id: Paper/document ID
            
        Returns:
            Partition key string
        """
        return f"DOC#{paper_id}"
    
    def get_errors_for_paper_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get errors from DynamoDB for specific paper IDs.
        
        Args:
            paper_ids: List of paper IDs to get errors for
            
        Returns:
            List of error dictionaries with paper_id included. Each dict contains:
            - 'paper_id': The paper ID
            - 'timestamp': Error timestamp
            - 'error_data': Dictionary with error details (stderr, stdout, error_message, etc.)
        """
        if not self.client or not self.resource:
            logger.warning("DynamoDB client not available - cannot retrieve errors for similar papers")
            return []
        
        all_errors = []
        try:
            table = self.resource.Table(self.table_name)
            
            for paper_id in paper_ids:
                try:
                    # Build partition key: DOC#<paper_id>
                    partition_key = self._get_partition_key(paper_id)
                    
                    # Query DynamoDB by partition key
                    response = table.query(
                        KeyConditionExpression='docID = :pk',
                        ExpressionAttributeValues={
                            ':pk': partition_key
                        },
                        ScanIndexForward=True  # Sort ascending by sort key (iteration order)
                    )
                    
                    # Parse errors
                    for item in response.get('Items', []):
                        error_data_str = item.get('error_data', '{}')
                        try:
                            error_data = json.loads(error_data_str) if isinstance(error_data_str, str) else error_data_str
                        except json.JSONDecodeError:
                            # Fallback: reconstruct from individual fields
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
            
            logger.info(f"Retrieved {len(all_errors)} errors from {len(paper_ids)} papers")
            return all_errors
            
        except Exception as e:
            logger.error(f"Error retrieving errors from DynamoDB: {e}")
            return []
    
    def get_errors_for_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get errors for a single paper ID.
        
        Args:
            paper_id: Paper/document ID
            
        Returns:
            List of error dictionaries for the paper
        """
        return self.get_errors_for_paper_ids([paper_id])
    
    def get_error_count(self, paper_id: str) -> int:
        """
        Get the count of errors for a specific paper.
        
        Args:
            paper_id: Paper/document ID
            
        Returns:
            Number of errors for the paper
        """
        if not self.client or not self.resource:
            logger.warning("DynamoDB client not available - cannot get error count")
            return 0
        
        try:
            table = self.resource.Table(self.table_name)
            partition_key = self._get_partition_key(paper_id)
            
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
            
        except Exception as e:
            logger.error(f"Error getting error count for {paper_id}: {e}")
            return 0

