"""
Randomly sample papers from OpenSearch and generate code for them.

This script randomly selects 5 papers from OpenSearch and generates
PyTorch code for each one, saving results to the generated_code directory.
"""

import os
import sys
import random
import logging
from typing import List, Dict, Any

# Add parent directory to path to import code_gen modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_gen.opensearch_client import OpenSearchClient
from test_code_generation import LocalCodeGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_random_paper_ids(count: int = 5, exclude_code_generated: bool = False) -> List[str]:
    """
    Get random paper IDs from OpenSearch.
    
    Args:
        count: Number of paper IDs to return
        exclude_code_generated: If True, exclude papers that already have code generated
        
    Returns:
        List of paper IDs
    """
    try:
        opensearch_client = OpenSearchClient()
        
        # Build query
        if exclude_code_generated:
            query = {
                "bool": {
                    "must_not": [
                        {"exists": {"field": "code_generated"}}
                    ]
                }
            }
        else:
            query = {"match_all": {}}
        
        # Get all papers (or papers without code)
        papers = opensearch_client.search_papers(query, size=1000)  # Get up to 1000 papers
        
        if not papers:
            logger.warning("No papers found in OpenSearch")
            return []
        
        # Extract paper IDs
        paper_ids = [paper.get('_id') for paper in papers if paper.get('_id')]
        
        if len(paper_ids) < count:
            logger.warning(f"Only found {len(paper_ids)} papers, returning all of them")
            return paper_ids
        
        # Randomly sample
        sampled_ids = random.sample(paper_ids, count)
        logger.info(f"Randomly sampled {len(sampled_ids)} papers from {len(paper_ids)} total papers")
        
        return sampled_ids
        
    except Exception as e:
        logger.error(f"Error getting random paper IDs: {e}")
        return []


def generate_code_for_papers(paper_ids: List[str], output_dir: str = "generated_code", 
                            clear_dir: bool = True, include_full_content: bool = False) -> Dict[str, Any]:
    """
    Generate code for multiple papers.
    
    Args:
        paper_ids: List of paper IDs to generate code for
        output_dir: Directory to save generated code
        clear_dir: Whether to clear output directory before saving
        include_full_content: Whether to include full paper content in generation
        
    Returns:
        Dictionary with results summary
    """
    generator = LocalCodeGenerator()
    
    results = {
        "total_papers": len(paper_ids),
        "successful": 0,
        "failed": 0,
        "results": []
    }
    
    # Clear directory if requested
    if clear_dir and os.path.exists(output_dir):
        logger.info(f"Clearing directory: {output_dir}")
        import shutil
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
    
    # Generate code for each paper
    for idx, paper_id in enumerate(paper_ids, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing paper {idx}/{len(paper_ids)}: {paper_id}")
        logger.info(f"{'='*80}")
        
        try:
            result = generator.generate_from_paper_id(
                paper_id,
                include_full_content=include_full_content
            )
            
            if result.get('success'):
                # Save code
                saved_path = generator.save_code_locally(result, output_dir, clear_dir=False)
                if saved_path:
                    logger.info(f"✓ Successfully generated and saved code for: {result.get('paper_title', 'Unknown')}")
                    results["successful"] += 1
                else:
                    logger.error(f"✗ Failed to save code for paper {paper_id}")
                    results["failed"] += 1
            else:
                logger.error(f"✗ Code generation failed: {result.get('error', 'Unknown error')}")
                results["failed"] += 1
            
            results["results"].append({
                "paper_id": paper_id,
                "success": result.get('success', False),
                "paper_title": result.get('paper_title', 'Unknown'),
                "error": result.get('error') if not result.get('success') else None
            })
            
        except Exception as e:
            logger.error(f"✗ Error processing paper {paper_id}: {e}")
            results["failed"] += 1
            results["results"].append({
                "paper_id": paper_id,
                "success": False,
                "paper_title": "Unknown",
                "error": str(e)
            })
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Randomly sample papers from OpenSearch and generate code for them',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--count', type=int, default=5,
                       help='Number of papers to randomly sample (default: 5)')
    parser.add_argument('--exclude-code-generated', action='store_true',
                       help='Exclude papers that already have code generated')
    parser.add_argument('--output-dir', default='generated_code',
                       help='Output directory for generated code (default: generated_code)')
    parser.add_argument('--no-clear', action='store_true',
                       help='Do not clear output directory before saving')
    parser.add_argument('--include-full-content', action='store_true',
                       help='Include full paper content in generation')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("RANDOM PAPER CODE GENERATION")
    print("="*80)
    print(f"Sampling {args.count} papers from OpenSearch...")
    if args.exclude_code_generated:
        print("Excluding papers with existing code...")
    print("="*80 + "\n")
    
    # Get random paper IDs
    paper_ids = get_random_paper_ids(
        count=args.count,
        exclude_code_generated=args.exclude_code_generated
    )
    
    if not paper_ids:
        print("❌ No papers found. Exiting.")
        return
    
    print(f"\n📋 Selected {len(paper_ids)} papers:")
    for idx, paper_id in enumerate(paper_ids, 1):
        print(f"   {idx}. {paper_id}")
    print()
    
    # Generate code
    results = generate_code_for_papers(
        paper_ids=paper_ids,
        output_dir=args.output_dir,
        clear_dir=not args.no_clear,
        include_full_content=args.include_full_content
    )
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    print(f"Total papers processed: {results['total_papers']}")
    print(f"✓ Successful: {results['successful']}")
    print(f"✗ Failed: {results['failed']}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Print detailed results
    if results['results']:
        print("\nDetailed Results:")
        print("-"*80)
        for result in results['results']:
            status = "✓" if result['success'] else "✗"
            print(f"{status} {result['paper_id']}: {result['paper_title']}")
            if not result['success'] and result.get('error'):
                print(f"   Error: {result['error']}")
        print("-"*80)
    
    print(f"\n✅ Generated code saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

