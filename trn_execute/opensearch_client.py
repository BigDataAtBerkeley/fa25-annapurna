import os
import json
import logging
from typing import Dict, List, Any, Optional
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

logger = logging.getLogger(__name__)

class OpenSearchClient:
    
    def __init__(self):
        """Initialize OpenSearch client with AWS auth"""
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.opensearch_endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        self.opensearch_index = os.getenv("OPENSEARCH_INDEX")
        
        if not self.opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT environment variable is required")
        
        session = boto3.Session(region_name=self.aws_region)
        creds = session.get_credentials().get_frozen_credentials()
        auth = AWSV4SignerAuth(creds, self.aws_region, "es")
        
        self.client = OpenSearch(
            hosts=[{"host": self.opensearch_endpoint.replace("https://", "").replace("http://", ""), "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60
        )
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.client.get(index=self.opensearch_index, id=paper_id)
            return response['_source']
        except Exception as e:
            logger.error(f"Error retrieving paper {paper_id}: {e}")
            return None
    
    def get_paper_summary(self, paper: Dict[str, Any]) -> str:
        title = paper.get('title', 'Unknown Title')
        authors = paper.get('authors', [])
        abstract = paper.get('abstract', 'No abstract available')
        date = paper.get('date', 'Unknown date')
        
        summary = f"""
                Paper Title: {title}
                Authors: {', '.join(authors) if isinstance(authors, list) else authors}
                Date: {date}
                Abstract: {abstract}
                """
        
        return summary.strip()
    
    def search_similar_papers(self, paper_id: str, exclude_id: str = None, size: int = 5) -> List[Dict[str, Any]]:
        try:
            paper = self.get_paper_by_id(paper_id)
            if not paper:
                logger.warning(f"Paper {paper_id} not found")
                return []
            
            embedding = paper.get("abstract_embedding")
            if not embedding:
                logger.warning(f"Paper {paper_id} has no abstract_embedding field")
                return []
                        
            query_size = size + 1 if exclude_id else size
            query_primary = {
                "knn": {
                    "abstract_embedding": {
                        "vector": embedding,
                        "k": query_size
                    }
                }
            }
        
            try:
                response = self.client.search(
                    index=self.opensearch_index,
                    body={"query": query_primary, "size": query_size}
                )
            except Exception as e1:
                logger.warning(f"Primary kNN query failed, retrying with field/query_vector: {e1}")
                query_fallback = {
                    "knn": {
                        "field": "abstract_embedding",
                        "query_vector": embedding,
                        "k": query_size
                    }
                }
                response = self.client.search(
                    index=self.opensearch_index,
                    body={"query": query_fallback, "size": query_size}
                )
            
            papers = []
            for hit in response.get('hits', {}).get('hits', []):
                doc_id = hit.get('_id')
                
                if exclude_id and doc_id == exclude_id:
                    continue
                
                paper = hit.get('_source', {})
                paper['_id'] = doc_id
                paper['similarity_score'] = hit.get('_score')
                papers.append(paper)
                
                if len(papers) >= size:
                    break
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            return []
    
    def update_paper_execution_results(self, paper_id: str, execution_results: Dict[str, Any]) -> bool:
        try:
            # Filter out None values
            filtered_results = {k: v for k, v in execution_results.items() if v is not None}
            
            # Update document
            self.client.update(
                index=self.opensearch_index,
                id=paper_id,
                body={
                    "doc": filtered_results,
                }
            )
            logger.info(f"Updated OpenSearch document {paper_id} with execution results")
            return True
        except Exception as e:
            logger.error(f"Error updating OpenSearch document {paper_id}: {e}")
            return False
