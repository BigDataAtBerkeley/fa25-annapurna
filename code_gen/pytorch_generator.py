"""
PyTorch code generator that orchestrates the entire process.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
try:
    from opensearch_client import OpenSearchClient
    from bedrock_client import BedrockClient
    from dataset_recommender import DatasetRecommender
except ImportError:
    from .opensearch_client import OpenSearchClient
    from .bedrock_client import BedrockClient
    from .dataset_recommender import DatasetRecommender

logger = logging.getLogger(__name__)

class PyTorchCodeGenerator:
    """Main class that orchestrates the PyTorch code generation process."""
    
    def __init__(self):
        """initilaize code generator"""
        self.opensearch_client = OpenSearchClient()
        self.bedrock_client = BedrockClient()
        self.dataset_recommender = DatasetRecommender(bedrock_client=self.bedrock_client)
        
        logger.info("PyTorch Code Generator initialized")
    
    def generate_code_for_paper(self, paper_id: str, include_full_content: bool = False) -> Dict[str, Any]:
        """
        Generate PyTorch code for a single paper by ID.
        
        Args:
            paper_id: OpenSearch document ID of the paper
            include_full_content: Whether to include full paper content in generation
            
        Returns:
            Dictionary containing generated code and metadata
        """
        try:
            logger.info(f"Generating code for paper ID: {paper_id}")
            
            # Retrieve paper from OpenSearch
            paper = self.opensearch_client.get_paper_by_id(paper_id)
            if not paper:
                return {
                    "success": False,
                    "error": f"Paper with ID {paper_id} not found",
                    "paper_id": paper_id
                }
            
            # Guard: Skip if code already generated for this paper
            if paper.get("code_generated") is True or (
                paper.get("code_s3_bucket") and paper.get("code_s3_key")
            ):
                return {
                    "success": True,
                    "already_generated": True,
                    "paper_id": paper_id,
                    "paper_title": paper.get('title', 'Unknown'),
                    "paper_authors": paper.get('authors', []),
                    "generated_at": datetime.now().isoformat()
                }
            
            # Get paper summary
            paper_summary = self.opensearch_client.get_paper_summary(paper)
            
            # Optionally get full paper content
            paper_content = None
            if include_full_content:
                paper_content = self.opensearch_client.get_paper_content(paper)
            
            # Get dataset recommendations
            dataset_recommendations = self.dataset_recommender.recommend_datasets(
                paper, paper_content, use_llm=True
            )
            logger.info(f"Recommended datasets: {dataset_recommendations.get('recommended_datasets', [])}")
            
            # Generate PyTorch code using Bedrock with dataset recommendations
            result = self.bedrock_client.generate_pytorch_code(
                paper_summary, 
                paper_content,
                dataset_recommendations=dataset_recommendations
            )
            
            # Add metadata including dataset recommendations
            result.update({
                "paper_id": paper_id,
                "paper_title": paper.get('title', 'Unknown'),
                "paper_authors": paper.get('authors', []),
                "generated_at": datetime.now().isoformat(),
                "include_full_content": include_full_content,
                "dataset_recommendations": dataset_recommendations,
                "recommended_dataset": dataset_recommendations.get("primary_dataset", "synthetic")
            })
            
            logger.info(f"Successfully generated code for paper: {paper.get('title', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating code for paper {paper_id}: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "paper_id": paper_id
            }
    
    def generate_code_for_papers(self, paper_ids: List[str], include_full_content: bool = False) -> Dict[str, Any]:
        """
        Generate PyTorch code for multiple papers.
        
        Args:
            paper_ids: List of OpenSearch document IDs
            include_full_content: Whether to include full paper content in generation
            
        Returns:
            Dictionary containing results for all papers
        """
        results = {
            "success": True,
            "total_papers": len(paper_ids),
            "successful_generations": 0,
            "failed_generations": 0,
            "results": [],
            "generated_at": datetime.now().isoformat()
        }
        
        for paper_id in paper_ids:
            result = self.generate_code_for_paper(paper_id, include_full_content)
            results["results"].append(result)
            
            if result["success"]:
                results["successful_generations"] += 1
            else:
                results["failed_generations"] += 1
        
        logger.info(f"Generated code for {results['successful_generations']}/{results['total_papers']} papers")
        return results
    
    def search_and_generate_code(self, search_query: Dict[str, Any], max_papers: int = 5, 
                               include_full_content: bool = False) -> Dict[str, Any]:
        """
        Search for papers and generate code for the results.
        
        Args:
            search_query: OpenSearch query DSL
            max_papers: Maximum number of papers to process
            include_full_content: Whether to include full paper content in generation
            
        Returns:
            Dictionary containing search results and generated code
        """
        try:
            logger.info(f"Searching for papers with query: {search_query}")
            
            # Search for papers
            papers = self.opensearch_client.search_papers(search_query, max_papers)
            
            if not papers:
                return {
                    "success": False,
                    "error": "No papers found matching the search criteria",
                    "search_query": search_query
                }
            
            # Extract paper IDs
            paper_ids = [paper.get('_id') for paper in papers if paper.get('_id')]
            
            # Generate code for found papers
            results = self.generate_code_for_papers(paper_ids, include_full_content)
            
            # Add search metadata
            results.update({
                "search_query": search_query,
                "papers_found": len(papers),
                "papers_processed": len(paper_ids)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search and generate: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "search_query": search_query
            }
    
    def generate_code_by_title(self, title: str, max_papers: int = 3, 
                              include_full_content: bool = False) -> Dict[str, Any]:
        """
        Generate code for papers matching a title.
        
        Args:
            title: Paper title to search for
            max_papers: Maximum number of papers to process
            include_full_content: Whether to include full paper content in generation
            
        Returns:
            Dictionary containing generated code and metadata
        """
        search_query = {"match": {"title": title}}
        return self.search_and_generate_code(search_query, max_papers, include_full_content)
    
    def generate_code_by_author(self, author: str, max_papers: int = 5, 
                              include_full_content: bool = False) -> Dict[str, Any]:
        """
        Generate code for papers by a specific author.
        
        Args:
            author: Author name to search for
            max_papers: Maximum number of papers to process
            include_full_content: Whether to include full paper content in generation
            
        Returns:
            Dictionary containing generated code and metadata
        """
        search_query = {"match": {"authors": author}}
        return self.search_and_generate_code(search_query, max_papers, include_full_content)
    
    def generate_code_by_keywords(self, keywords: str, max_papers: int = 5, 
                                 include_full_content: bool = False) -> Dict[str, Any]:
        """
        Generate code for papers matching abstract keywords.
        
        Args:
            keywords: Keywords to search in abstract
            max_papers: Maximum number of papers to process
            include_full_content: Whether to include full paper content in generation
            
        Returns:
            Dictionary containing generated code and metadata
        """
        search_query = {"match": {"abstract": keywords}}
        return self.search_and_generate_code(search_query, max_papers, include_full_content)
    
    def generate_code_for_recent_papers(self, days: int = 30, max_papers: int = 10, 
                                      include_full_content: bool = False) -> Dict[str, Any]:
        """
        Generate code for recently ingested papers.
        
        Args:
            days: Number of days to look back
            max_papers: Maximum number of papers to process
            include_full_content: Whether to include full paper content in generation
            
        Returns:
            Dictionary containing generated code and metadata
        """
        search_query = {
            "range": {
                "ingested_at": {
                    "gte": f"now-{days}d"
                }
            }
        }
        return self.search_and_generate_code(search_query, max_papers, include_full_content)
    
    def save_generated_code(self, result: Dict[str, Any], output_dir: str = "generated_code") -> str:
        """
        Save generated code to files.
        
        Args:
            result: Result from code generation
            output_dir: Directory to save files (default: generated_code)
            
        Returns:
            Path to saved file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            if not result.get("success"):
                logger.error(f"Cannot save failed generation: {result.get('error')}")
                return None
            
            # Generate filename
            paper_title = result.get("paper_title", "unknown_paper")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Clean filename
            safe_title = "".join(c for c in paper_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
            
            filename = f"{safe_title}_{timestamp}.py"
            filepath = os.path.join(output_dir, filename)
            
            # Save code with header
            with open(filepath, 'w', encoding='utf-8') as f:
                # Add paper metadata as header
                f.write(f'"""\n')
                f.write(f'{result.get("paper_title", "Unknown Title")}\n\n')
                f.write(f'Generated by AWS Bedrock Claude\n')
                f.write(f'Paper ID: {result.get("paper_id", "N/A")}\n')
                authors = result.get("paper_authors", [])
                if isinstance(authors, list):
                    f.write(f'Authors: {", ".join(authors)}\n')
                else:
                    f.write(f'Authors: {authors}\n')
                f.write(f'Generated at: {result.get("generated_at", "N/A")}\n')
                f.write(f'"""\n\n')
                f.write(result["code"])
            
            metadata_file = filepath.replace('.py', '_metadata.json')
            metadata = {
                "paper_id": result.get("paper_id"),
                "paper_title": result.get("paper_title"),
                "paper_authors": result.get("paper_authors", []),
                "explanation": result.get("explanation"),  # Contains key info about metrics/improvements
                "generated_at": result.get("generated_at"),
                "model_used": result.get("model_used"),
                "code_file": filepath
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved generated code to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving generated code: {e}")
            return None
    
    def get_paper_info(self, paper_id: str) -> Dict[str, Any]:
        """
        Get information about a paper without generating code.
        
        Args:
            paper_id: OpenSearch document ID
            
        Returns:
            Paper information dictionary
        """
        try:
            paper = self.opensearch_client.get_paper_by_id(paper_id)
            if not paper:
                return {
                    "success": False,
                    "error": f"Paper with ID {paper_id} not found"
                }
            
            return {
                "success": True,
                "paper": paper,
                "summary": self.opensearch_client.get_paper_summary(paper)
            }
            
        except Exception as e:
            logger.error(f"Error getting paper info: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}"
            }
