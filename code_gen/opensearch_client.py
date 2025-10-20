"""
OpenSearch client for retrieving papers and performing vector similarity search.
Enhanced with RAG capabilities for paper deduplication.
"""

import os 
import json
import logging
from typing import Dict, List, Optional, Any
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from dotenv import load_dotenv
from .embeddings_client import EmbeddingsClient

load_dotenv()

logger = logging.getLogger(__name__)

class OpenSearchClient:
    """Client for interacting with OpenSearch to retrieve research papers with RAG capabilities"""
    
    def __init__(self):
        """Initializes OpenSearch client with AWS auth and embeddings support"""
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
        
        # Initialize embeddings client for RAG
        self.embeddings_client = EmbeddingsClient()
        
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
            content = response['Body'].read().decode('utf-8')
            
            logger.info(f"Retrieved paper content from s3://{s3_bucket}/{s3_key}")
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
    
    def ensure_vector_index(self) -> bool:
        """
        Ensure the OpenSearch index has proper mapping for vector search.
        Creates the index with k-NN mapping if it doesn't exist.
        
        Returns:
            True if index is ready for vector search, False otherwise
        """
        try:
            if not self.client.indices.exists(index=self.opensearch_index):
                # Create index with vector mapping
                mapping = {
                    "settings": {
                        "index": {
                            "number_of_shards": 1,
                            "knn": True,
                            "knn.algo_param.ef_search": 100
                        }
                    },
                    "mappings": {
                        "properties": {
                            "title": {"type": "text"},
                            "title_normalized": {"type": "keyword"},
                            "authors": {"type": "keyword"},
                            "abstract": {"type": "text"},
                            "abstract_embedding": {
                                "type": "knn_vector",
                                "dimension": 1536,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib",
                                    "parameters": {
                                        "ef_construction": 128,
                                        "m": 24
                                    }
                                }
                            },
                            "date": {"type": "date"},
                            "s3_bucket": {"type": "keyword"},
                            "s3_key": {"type": "keyword"},
                            "sha_abstract": {"type": "keyword"},
                            "decision": {"type": "keyword"},
                            "reason": {"type": "text"},
                            "relevance": {"type": "keyword"},
                            "novelty": {"type": "keyword"},
                            "ingested_at": {"type": "date"}
                        }
                    }
                }
                
                self.client.indices.create(index=self.opensearch_index, body=mapping)
                logger.info(f"Created OpenSearch index {self.opensearch_index} with vector mapping")
                return True
            else:
                logger.info(f"OpenSearch index {self.opensearch_index} already exists")
                return True
                
        except Exception as e:
            logger.error(f"Error ensuring vector index: {e}")
            return False
    
    def search_similar_papers(self, abstract: str, size: int = 5, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search for papers with similar abstracts using vector similarity.
        
        Args:
            abstract: Abstract text to search for similar papers
            size: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of similar papers with similarity scores
        """
        try:
            # Generate embedding for the abstract
            embedding = self.embeddings_client.generate_embedding(abstract)
            if not embedding:
                logger.error("Failed to generate embedding for abstract")
                return []
            
            # Perform k-NN search
            query = {
                "knn": {
                    "field": "abstract_embedding",
                    "query_vector": embedding,
                    "k": size,
                    "num_candidates": 100
                }
            }
            
            response = self.client.search(
                index=self.opensearch_index,
                body={"query": query, "size": size}
            )
            
            papers = []
            for hit in response['hits']['hits']:
                paper = hit['_source']
                paper['_id'] = hit['_id']
                paper['_score'] = hit['_score']
                paper['similarity_score'] = hit['_score']  # Cosine similarity score
                
                # Filter by similarity threshold
                if hit['_score'] >= similarity_threshold:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} similar papers above threshold {similarity_threshold}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            return []
    
    def index_paper_with_embedding(self, paper_data: Dict[str, Any]) -> bool:
        """
        Index a paper with its abstract embedding.
        
        Args:
            paper_data: Paper data including title, abstract, authors, etc.
            
        Returns:
            True if successfully indexed, False otherwise
        """
        try:
            # Generate embedding for abstract
            abstract = paper_data.get('abstract', '')
            if not abstract:
                logger.warning("No abstract provided, cannot generate embedding")
                return False
            
            embedding = self.embeddings_client.generate_embedding(abstract)
            if not embedding:
                logger.error("Failed to generate embedding for paper")
                return False
            
            # Add embedding to paper data
            paper_data['abstract_embedding'] = embedding
            
            # Index the paper
            paper_id = paper_data.get('_id', paper_data.get('title_normalized', ''))
            self.client.index(
                index=self.opensearch_index,
                id=paper_id,
                body=paper_data,
                refresh=True
            )
            
            logger.info(f"Successfully indexed paper with embedding: {paper_data.get('title', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing paper with embedding: {e}")
            return False
    
    def check_paper_redundancy(self, title: str, abstract: str, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Check if a paper is redundant based on vector similarity to existing papers.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            similarity_threshold: Threshold for considering papers similar
            
        Returns:
            Dictionary with redundancy check results
        """
        try:
            # Search for similar papers
            similar_papers = self.search_similar_papers(abstract, size=10, similarity_threshold=similarity_threshold)
            
            if not similar_papers:
                return {
                    "is_redundant": False,
                    "similar_papers": [],
                    "max_similarity": 0.0,
                    "reason": "No similar papers found"
                }
            
            # Get the most similar paper
            most_similar = similar_papers[0]
            max_similarity = most_similar.get('similarity_score', 0.0)
            
            # Determine if redundant based on similarity threshold
            is_redundant = max_similarity >= similarity_threshold
            
            return {
                "is_redundant": is_redundant,
                "similar_papers": similar_papers[:5],  # Top 5 similar papers
                "max_similarity": max_similarity,
                "reason": f"Most similar paper has {max_similarity:.3f} similarity score" if is_redundant else "Similarity below threshold"
            }
            
        except Exception as e:
            logger.error(f"Error checking paper redundancy: {e}")
            return {
                "is_redundant": False,
                "similar_papers": [],
                "max_similarity": 0.0,
                "reason": f"Error during redundancy check: {str(e)}"
            }
