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
        self.opensearch_index = os.getenv("OPENSEARCH_INDEX", "research-papers-v3")
        
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
    
    def get_random_papers(self, size: int = 10) -> List[Dict[str, Any]]:
        """
        Get random papers from the index.
        
        Args:
            size: Maximum number of results
            
        Returns:
            List of random papers
        """
        # Use function_score with random_score to get random papers
        query = {
            "function_score": {
                "query": {"match_all": {}},
                "random_score": {},
                "boost_mode": "replace"
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
            
            # Check if it's a PDF - if so, return None (use get_paper_pdf_bytes instead)
            if s3_key.lower().endswith('.pdf'):
                logger.info(f"Paper is a PDF file - use get_paper_pdf_bytes() for PDF processing")
                return None
            
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
    
    def get_paper_pdf_bytes(self, paper: Dict[str, Any]) -> Optional[bytes]:
        """
        Retrieve the PDF file bytes from S3 (for PDF processing).
        
        Args:
            paper: Paper document from OpenSearch
            
        Returns:
            PDF file as bytes or None if not available
        """
        try:
            s3_bucket = paper.get('s3_bucket')
            s3_key = paper.get('s3_key')
            
            if not s3_bucket or not s3_key:
                logger.warning("Paper missing S3 bucket or key information")
                return None
            
            # Initialize S3 client
            s3_client = boto3.client('s3', region_name=self.aws_region)
            
            # Download the PDF file
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            pdf_bytes = response['Body'].read()
            
            logger.info(f"Retrieved PDF from s3://{s3_bucket}/{s3_key} ({len(pdf_bytes)} bytes)")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Error retrieving PDF from S3: {e}")
            return None
    
    def is_pdf_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Check if the paper is a PDF file.
        
        Args:
            paper: Paper document from OpenSearch
            
        Returns:
            True if paper is a PDF
        """
        s3_key = paper.get('s3_key', '')
        return s3_key.lower().endswith('.pdf')
    
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
    
    def search_similar_papers_by_abstract(self, paper_id: str, exclude_id: str = None, size: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar papers using abstract embedding (KNN search).
        Uses the paper's existing embedding from OpenSearch.
        
        Args:
            paper_id: Paper ID to get embedding from
            exclude_id: Paper ID to exclude from results (optional)
            size: Number of similar papers to return (default 5)
            
        Returns:
            List of similar paper documents with similarity scores
        """
        try:
            # Get paper and its existing embedding
            paper = self.get_paper_by_id(paper_id)
            if not paper:
                logger.warning(f"Paper {paper_id} not found")
                return []
            
            embedding = paper.get("abstract_embedding")
            if not embedding:
                logger.warning(f"Paper {paper_id} has no abstract_embedding field")
                return []
            
            # Convert to float list
            try:
                embedding = [float(x) for x in embedding]
            except Exception:
                logger.warning(f"Failed to convert embedding to float list")
                return []
            
            # Get embedding dimension from index mapping
            try:
                mapping = self.client.indices.get_mapping(index=self.opensearch_index)
                props = mapping.get(self.opensearch_index, {}).get("mappings", {}).get("properties", {})
                dim = int(props.get("abstract_embedding", {}).get("dimension", len(embedding)))
            except Exception:
                dim = len(embedding)
            
            # Adjust embedding dimension if needed
            if len(embedding) > dim:
                embedding = embedding[:dim]
            elif len(embedding) < dim:
                embedding = embedding + [0.0] * (dim - len(embedding))
            
            # Build KNN query
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
            
            # Process results
            papers = []
            for hit in response.get('hits', {}).get('hits', []):
                doc_id = hit.get('_id')
                
                # Skip the excluded document (self-comparison guard)
                if exclude_id and doc_id == exclude_id:
                    logger.info(f"Skipping self-match: {doc_id}")
                    continue
                
                paper = hit.get('_source', {})
                paper['_id'] = doc_id
                paper['similarity_score'] = hit.get('_score')
                papers.append(paper)
                
                # Stop once we have enough results
                if len(papers) >= size:
                    break
            
            logger.info(f"Found {len(papers)} similar papers (excluded: {exclude_id})")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []