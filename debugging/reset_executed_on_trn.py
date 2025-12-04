#!/usr/bin/env python3
"""
Reset executed_on_trn field to false for all papers in OpenSearch.
This allows the cron job to pick them up again for processing.
"""

import os
import json
import logging
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT", "search-research-papers-uv3fxq76j5bkxq3bgp3nyfdtnm.us-east-1.es.amazonaws.com")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v3")

# Initialize OpenSearch client
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

def get_all_papers(size: int = 1000) -> List[Dict[str, Any]]:
    """Get all papers from OpenSearch."""
    try:
        response = os_client.search(
            index=OPENSEARCH_INDEX,
            body={
                "query": {"match_all": {}},
                "size": size
            }
        )
        
        papers = []
        for hit in response.get('hits', {}).get('hits', []):
            paper = hit['_source']
            paper['_id'] = hit['_id']
            papers.append(paper)
        
        total = response.get('hits', {}).get('total', {})
        if isinstance(total, dict):
            total_count = total.get('value', len(papers))
        else:
            total_count = total
        
        logger.info(f"Found {len(papers)} papers (total in index: {total_count})")
        return papers, total_count
        
    except Exception as e:
        logger.error(f"Error querying OpenSearch: {e}")
        return [], 0

def reset_executed_on_trn_bulk(papers: List[Dict[str, Any]]) -> tuple[int, int]:
    """Bulk update executed_on_trn to false for all papers."""
    from opensearchpy.helpers import bulk
    
    updated = 0
    failed = 0
    
    # Prepare bulk update actions
    actions = []
    for paper in papers:
        paper_id = paper.get('_id')
        current_value = paper.get('executed_on_trn')
        
        # Only update if not already False
        if current_value is not False:
            actions.append({
                "_op_type": "update",
                "_index": OPENSEARCH_INDEX,
                "_id": paper_id,
                "doc": {
                    "executed_on_trn": False
                }
            })
        else:
            updated += 1  # Already False, count as updated
    
    if not actions:
        logger.info("All papers already have executed_on_trn = False")
        return updated, failed
    
    logger.info(f"Bulk updating {len(actions)} papers...")
    
    # Execute bulk update
    try:
        success, failed_items = bulk(os_client, actions, chunk_size=50, request_timeout=60)
        updated += success
        failed += len(failed_items) if failed_items else 0
        
        if failed_items:
            logger.warning(f"Failed to update {len(failed_items)} papers")
            for item in failed_items[:5]:  # Show first 5 failures
                logger.warning(f"  Failed: {item.get('update', {}).get('_id', 'unknown')}")
        
        return updated, failed
    except Exception as e:
        logger.error(f"Bulk update failed: {e}")
        return updated, len(actions)

def main():
    logger.info(f"Resetting executed_on_trn to false for all papers in {OPENSEARCH_INDEX}")
    logger.info(f"OpenSearch endpoint: {OPENSEARCH_ENDPOINT}")
    
    # Get all papers
    papers, total_count = get_all_papers(size=1000)
    
    if not papers:
        logger.warning("No papers found in OpenSearch")
        return
    
    logger.info(f"Found {len(papers)} papers to check")
    
    # Bulk update all papers
    updated, failed = reset_executed_on_trn_bulk(papers)
    
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"✅ Reset complete!")
    logger.info(f"   Total papers: {len(papers)}")
    logger.info(f"   Updated: {updated}")
    logger.info(f"   Failed: {failed}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()

