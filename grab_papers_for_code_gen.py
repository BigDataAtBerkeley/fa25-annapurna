#!/usr/bin/env python3
"""
Grab 3 papers from OpenSearch and send them to code-evaluation queue.

This script:
1. Queries OpenSearch for papers (optionally resets code_generated flag if needed)
2. Selects 3 papers
3. Sends them to code-evaluation.fifo SQS queue

Usage:
    python grab_papers_for_code_gen.py [--reset-existing]
    
Options:
    --reset-existing: Reset code_generated flag for papers that already have code
"""

import os
import sys
import json
import logging
import boto3
import argparse
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add code_gen to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code_gen'))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CODE_EVAL_QUEUE_NAME = "code-evaluation.fifo"
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v2")

def get_opensearch_client():
    """Get OpenSearch client"""
    from opensearch_client import OpenSearchClient
    return OpenSearchClient()

def get_code_eval_queue_url() -> str:
    """Get the code-evaluation queue URL"""
    sqs = boto3.client('sqs', region_name=AWS_REGION)
    account_id = boto3.client('sts').get_caller_identity()['Account']
    queue_url = f"https://sqs.{AWS_REGION}.amazonaws.com/{account_id}/{CODE_EVAL_QUEUE_NAME}"
    return queue_url

def reset_code_generated_flag(opensearch_client, paper_id: str) -> bool:
    """
    Reset code_generated flag for a paper (similar to reset_code_generated.py).
    
    Args:
        opensearch_client: OpenSearchClient instance
        paper_id: Paper ID to reset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from opensearchpy import OpenSearch
        # Get the OpenSearch client from the client
        os_client = opensearch_client.client
        
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
        
        os_client.update(
            index=opensearch_client.opensearch_index or OPENSEARCH_INDEX,
            id=paper_id,
            body=update_body,
            refresh=True
        )
        
        logger.info(f"✅ Reset code_generated flag for paper {paper_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to reset code_generated for {paper_id}: {e}")
        return False

def get_papers_without_code(opensearch_client, limit: int = 3, reset_existing: bool = False) -> List[Dict[str, Any]]:
    """
    Get papers from OpenSearch. Optionally reset code_generated flag if papers already have code.
    
    Args:
        opensearch_client: OpenSearchClient instance
        limit: Number of papers to retrieve
        reset_existing: If True, reset code_generated flag for papers that already have code
        
    Returns:
        List of paper documents
    """
    # First, try to get papers without code
    query = {
        "bool": {
            "must_not": [
                {"term": {"code_generated": True}}
            ]
        }
    }
    
    papers = opensearch_client.search_papers(query, size=limit)
    
    # If we don't have enough papers and reset_existing is True, get papers with code and reset them
    if len(papers) < limit and reset_existing:
        logger.info(f"Only found {len(papers)} papers without code. Searching for papers with code to reset...")
        
        # Get papers that DO have code_generated
        query_with_code = {
            "term": {"code_generated": True}
        }
        
        papers_with_code = opensearch_client.search_papers(query_with_code, size=limit - len(papers))
        
        # Reset code_generated flag for these papers
        for paper in papers_with_code:
            paper_id = paper.get('_id') or paper.get('paper_id')
            if reset_code_generated_flag(opensearch_client, paper_id):
                papers.append(paper)
                if len(papers) >= limit:
                    break
    
    return papers[:limit]

def send_paper_to_queue(paper: Dict[str, Any], queue_url: str) -> bool:
    """
    Send a paper to the code-evaluation queue.
    
    Args:
        paper: Paper document from OpenSearch
        queue_url: SQS queue URL
        
    Returns:
        True if successful, False otherwise
    """
    try:
        sqs = boto3.client('sqs', region_name=AWS_REGION)
        paper_id = paper.get('_id') or paper.get('paper_id')
        paper_title = paper.get('title', 'Unknown')
        
        message = {
            "paper_id": paper_id,
            "action": "generate_by_id",
            "paper_title": paper_title,
            "queued_at": datetime.now().isoformat()
        }
        
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
            MessageGroupId=paper_id,
            MessageDeduplicationId=f"{paper_id}-{int(datetime.now().timestamp() * 1000)}"
        )
        
        logger.info(f"✅ Sent paper '{paper_title[:60]}...' ({paper_id}) to code-evaluation queue")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send paper {paper.get('_id')} to queue: {e}")
        return False

def main():
    """Main function to grab papers and send to queue"""
    parser = argparse.ArgumentParser(description='Grab papers from OpenSearch and send to code-evaluation queue')
    parser.add_argument(
        '--reset-existing',
        action='store_true',
        help='Reset code_generated flag for papers that already have code generated'
    )
    args = parser.parse_args()
    
    logger.info("🚀 Starting: Grab 3 papers from OpenSearch and send to code-evaluation queue")
    if args.reset_existing:
        logger.info("🔄 Will reset code_generated flag for papers that already have code")
    
    try:
        # Get OpenSearch client
        logger.info("📚 Connecting to OpenSearch...")
        opensearch_client = get_opensearch_client()
        
        # Get papers (optionally reset existing ones)
        logger.info("🔍 Searching for papers...")
        papers = get_papers_without_code(opensearch_client, limit=3, reset_existing=args.reset_existing)
        
        if not papers:
            logger.warning("⚠️  No papers found")
            if not args.reset_existing:
                logger.info("💡 Try running with --reset-existing to reset papers that already have code")
            return
        
        logger.info(f"📄 Found {len(papers)} papers")
        
        # Get queue URL
        queue_url = get_code_eval_queue_url()
        logger.info(f"📤 Queue URL: {queue_url}")
        
        # Send each paper to queue
        success_count = 0
        for i, paper in enumerate(papers, 1):
            paper_title = paper.get('title', 'Unknown')
            logger.info(f"\n[{i}/{len(papers)}] Processing: {paper_title[:60]}...")
            
            if send_paper_to_queue(paper, queue_url):
                success_count += 1
        
        logger.info(f"\n✅ Successfully sent {success_count}/{len(papers)} papers to code-evaluation queue")
        logger.info("💡 The code generation Lambda will process these papers automatically")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()

