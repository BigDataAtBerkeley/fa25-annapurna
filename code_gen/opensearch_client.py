"""
OpenSearch client SOLELY for retrieving the papers from our index
"""

import os 
import json
import logging
from typing import Dict, List, Optional, Any
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class OpenSearchClient:
    """Client for interacting with OpenSearch to retrieve research papers"""
    
    def __init__(self):
        """Initializes OpenSearch client with AWS auth"""
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.opensearch_endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        self.opensearch_index = os.getenv("OPENSEARCH_INDEX", "research-papers")
        
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
            timeout=60 # defines timeout just to get OpenSearch client (realistically should only be like 5 seconds)
        )
        
        logger.info(f"OpenSearch client initialized for index: {self.opensearch_index}")
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific paper by its ID.
        
        Args:
            paper_id: The document ID in OpenSearch
            
        Returns:
            Dictionary of paper metadata or None if not found
        """
        try:
            response = self.client.get(index=self.opensearch_index, id=paper_id)
            return response['_source']
        except Exception as e:
            logger.error(f"Error retrieving paper {paper_id}: {e}")
            return None
    
    def search_papers(self, query: Dict[str, Any], size: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers using OpenSearch query
        
        Args:
            query: OpenSearch query DSL
            size: Maximum number of papers to return (default 10)
            
        Returns:
            List of paper documents (dictionaries)
        """
        try:
            response = self.client.search(
                index=self.opensearch_index,
                body={"query": query, "size": size}
            )
            
            papers = []
            for hit in response['hits']['hits']:
                paper = hit['_source']
                paper['_id'] = hit['_id']  # Includes document ID
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers in OpenSearch. Requested {size} papers based on query")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def get_papers_by_title(self, title: str, size: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers by title
        
        Args:
            title: Title to search for
            size: Maximum number of results
            
        Returns:
            List of matching papers (dictionaries)
        """
        query = {
            "match": {
                "title": title
            }
        }
        return self.search_papers(query, size)
    
    def get_papers_by_author(self, author: str, size: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers by author.
        
        Args:
            author: Author name to search for
            size: Maximum number of results
            
        Returns:
            List of matching papers (dictionaries)
        """
        query = {
            "match": {
                "authors": author
            }
        }
        return self.search_papers(query, size)
    
    def get_papers_by_abstract(self, abstract_keywords: str, size: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers by abstract keywords 
        
        Args:
            abstract_keywords: Keywords to search in abstract
            size: Maximum number of results
            
        Returns:
            List of matching papers (dictionaries)
        """
        query = {
            "match": {
                "abstract": abstract_keywords
            }
        }
        return self.search_papers(query, size)
    
    def get_recent_papers(self, days: int = 30, size: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently ingested papers (recency defined by # of days --> default 30 days)
        
        Args:
            days: Number of days to look back
            size: Maximum number of results
            
        Returns:
            List of recent papers
        """
        query = {
            "range": {
                "ingested_at": {
                    "gte": f"now-{days}d"
                }
            }
        }
        return self.search_papers(query, size)
    
    def get_all_papers(self, size: int = 50) -> List[Dict[str, Any]]:
        """
        Get all papers from the index.
        
        Args:
            size: Maximum number of results
            
        Returns:
            List of all papers
        """
        query = {"match_all": {}}
        return self.search_papers(query, size)
    
    def get_paper_content(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Retrieve the full paper content from S3.
        
        Args:
            paper: Paper document from OpenSearch
            
        Returns:
            Paper content as string or None if not available
        """
        try:
            s3_bucket = paper.get('s3_bucket')
            s3_key = paper.get('s3_key')
            
            if not s3_bucket or not s3_key:
                logger.warning("Paper missing S3 bucket or key information")
                return None
            
            # Initialize S3 client
            s3_client = boto3.client('s3', region_name=self.aws_region)
            
            # Download the paper content
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            raw_content = response['Body'].read()
            
            # Try multiple encodings to handle different file formats
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    content = raw_content.decode(encoding)
                    logger.info(f"Successfully decoded paper content using {encoding}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            # If all encodings fail, use utf-8 with error handling
            if content is None:
                logger.warning(f"Could not decode with standard encodings, using utf-8 with error handling")
                content = raw_content.decode('utf-8', errors='replace')
            
            logger.info(f"Retrieved paper content from s3://{s3_bucket}/{s3_key} ({len(content)} chars)")
            return content
            
        except Exception as e:
            logger.error(f"Error retrieving paper content: {e}")
            return None
    
    def get_paper_summary(self, paper: Dict[str, Any]) -> str:
        """
        Get a summary of the paper for code generation.
        
        Args:
            paper: Paper document from OpenSearch
            
        Returns:
            Formatted paper summary
        """
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