"""
Test script for generating code from papers in S3.

This script allows you to explicitly pass in a paper from S3 and generate
its code locally without saving to S3 or updating OpenSearch.

Usage:
    # From S3 bucket and key directly
    python code-gen-testing/test_code_generation.py --s3-bucket llm-research-papers --s3-key papers/paper123.pdf
    
    # From paper ID (fetches from OpenSearch, but only for metadata - generates locally)
    python code-gen-testing/test_code_generation.py --paper-id 6-j63JkBP8oloYi_8CJH
    
    # Include full paper content
    python code-gen-testing/test_code_generation.py --paper-id 6-j63JkBP8oloYi_8CJH --include-full-content
    
    # Clear generated_code directory before saving
    python code-gen-testing/test_code_generation.py --paper-id 6-j63JkBP8oloYi_8CJH --clear-dir
"""

import os
import sys
import json
import argparse
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
import boto3
from io import BytesIO

# Add parent directory to path to import code_gen modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_gen.bedrock_client import BedrockClient
from code_gen.dataset_recommender import DatasetRecommender
from code_gen.opensearch_client import OpenSearchClient

# Try to import PDF extraction libraries
try:
    import pypdf
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2 as pypdf
        PDF_EXTRACTION_AVAILABLE = True
    except ImportError:
        PDF_EXTRACTION_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalCodeGenerator:
    """Generate code locally without AWS integrations."""
    
    def __init__(self):
        """Initialize code generation components."""
        self.bedrock_client = BedrockClient()
        self.dataset_recommender = DatasetRecommender(bedrock_client=self.bedrock_client)
        self.s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        logger.info("Local Code Generator initialized")
    
    def generate_from_s3(self, s3_bucket: str, s3_key: str, 
                         include_full_content: bool = False,
                         paper_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate code from a paper stored in S3.
        
        Args:
            s3_bucket: S3 bucket name
            s3_key: S3 key (path to paper)
            include_full_content: Whether to include full paper content
            paper_metadata: Optional paper metadata (title, authors, abstract, etc.)
            
        Returns:
            Dictionary with generated code and metadata
        """
        try:
            logger.info(f"Downloading paper from s3://{s3_bucket}/{s3_key}")
            
            # Download paper content from S3
            response = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            content_bytes = response['Body'].read()
            
            # Check if file is a PDF
            is_pdf = s3_key.lower().endswith('.pdf') or content_bytes[:4] == b'%PDF'
            
            if is_pdf:
                # Extract text from PDF
                if not PDF_EXTRACTION_AVAILABLE:
                    raise ValueError(
                        "PDF file detected but PDF extraction library not available. "
                        "Please install pypdf: pip install pypdf\n"
                        "Alternatively, use --paper-id to fetch text content from OpenSearch."
                    )
                
                logger.info("Detected PDF file, extracting text...")
                paper_content = self._extract_text_from_pdf(content_bytes)
                logger.info(f"Extracted text from PDF ({len(paper_content)} chars)")
            else:
                # Assume it's a text file
                try:
                    paper_content = content_bytes.decode('utf-8')
                    logger.info(f"Downloaded text content ({len(paper_content)} chars)")
                except UnicodeDecodeError:
                    # Try other encodings
                    try:
                        paper_content = content_bytes.decode('latin-1')
                        logger.info(f"Downloaded text content (latin-1 encoding, {len(paper_content)} chars)")
                    except Exception as e:
                        raise ValueError(
                            f"Could not decode file content. File may be binary or use unsupported encoding. "
                            f"Error: {e}\n"
                            f"If this is a PDF, ensure pypdf is installed: pip install pypdf\n"
                            f"Alternatively, use --paper-id to fetch text content from OpenSearch."
                        )
            
            # Use provided metadata or create minimal metadata
            if paper_metadata:
                paper = paper_metadata
            else:
                # Create minimal paper structure
                paper = {
                    'title': os.path.basename(s3_key).replace('.pdf', '').replace('_', ' '),
                    'authors': [],
                    'abstract': paper_content[:500] if paper_content else 'No abstract available',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
            
            # Create paper summary
            paper_summary = self._create_paper_summary(paper)
            
            # Get dataset recommendations
            full_content = paper_content if include_full_content else None
            dataset_recommendations = self.dataset_recommender.recommend_datasets(
                paper, full_content, use_llm=True
            )
            logger.info(f"Recommended datasets: {dataset_recommendations.get('recommended_datasets', [])}")
            
            # Generate PyTorch code
            result = self.bedrock_client.generate_pytorch_code(
                paper_summary,
                full_content,
                dataset_recommendations=dataset_recommendations
            )
            
            # Add metadata
            result.update({
                "paper_id": paper.get('_id', 'local-test'),
                "paper_title": paper.get('title', 'Unknown'),
                "paper_authors": paper.get('authors', []),
                "generated_at": datetime.now().isoformat(),
                "include_full_content": include_full_content,
                "dataset_recommendations": dataset_recommendations,
                "recommended_dataset": dataset_recommendations.get("primary_dataset", "synthetic"),
                "s3_bucket": s3_bucket,
                "s3_key": s3_key
            })
            
            logger.info(f"Successfully generated code for: {paper.get('title', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating code from S3: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "s3_bucket": s3_bucket,
                "s3_key": s3_key
            }
    
    def generate_from_paper_id(self, paper_id: str, 
                               include_full_content: bool = False) -> Dict[str, Any]:
        """
        Generate code from a paper ID (fetches from OpenSearch for metadata only).
        
        Args:
            paper_id: OpenSearch document ID
            include_full_content: Whether to include full paper content
            
        Returns:
            Dictionary with generated code and metadata
        """
        try:
            logger.info(f"Fetching paper metadata for ID: {paper_id}")
            
            # Fetch paper from OpenSearch (only for metadata)
            opensearch_client = OpenSearchClient()
            paper = opensearch_client.get_paper_by_id(paper_id)
            
            if not paper:
                return {
                    "success": False,
                    "error": f"Paper with ID {paper_id} not found in OpenSearch",
                    "paper_id": paper_id
                }
            
            # Get S3 location
            s3_bucket = paper.get('s3_bucket')
            s3_key = paper.get('s3_key')
            
            if not s3_bucket or not s3_key:
                return {
                    "success": False,
                    "error": f"Paper {paper_id} does not have S3 bucket/key information",
                    "paper_id": paper_id
                }
            
            # Add paper ID to metadata
            paper['_id'] = paper_id
            
            # Generate code using S3 location
            return self.generate_from_s3(
                s3_bucket=s3_bucket,
                s3_key=s3_key,
                include_full_content=include_full_content,
                paper_metadata=paper
            )
            
        except Exception as e:
            logger.error(f"Error generating code from paper ID: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "paper_id": paper_id
            }
    
    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Extracted text as string
        """
        try:
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            extracted_text = '\n\n'.join(text_parts)
            
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from PDF. PDF may be image-based or corrupted.")
            
            return extracted_text
            
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {e}")
    
    def _create_paper_summary(self, paper: Dict[str, Any]) -> str:
        """Create a paper summary string."""
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
    
    def save_code_locally(self, result: Dict[str, Any], output_dir: str = "generated_code", clear_dir: bool = False) -> str:
        """
        Save generated code to local directory.
        
        Args:
            result: Result from code generation
            output_dir: Directory to save files (default: generated_code)
            clear_dir: Whether to clear the directory before saving (default: False)
            
        Returns:
            Path to saved code file
        """
        try:
            if not result.get("success"):
                logger.error(f"Cannot save failed generation: {result.get('error')}")
                return None
            
            # Clear directory if requested
            if clear_dir and os.path.exists(output_dir):
                logger.info(f"Clearing directory: {output_dir}")
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
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
                f.write(f'Generated by AWS Bedrock Claude (LOCAL TEST)\n')
                f.write(f'Paper ID: {result.get("paper_id", "N/A")}\n')
                authors = result.get("paper_authors", [])
                if isinstance(authors, list):
                    f.write(f'Authors: {", ".join(authors)}\n')
                else:
                    f.write(f'Authors: {authors}\n')
                f.write(f'Generated at: {result.get("generated_at", "N/A")}\n')
                if result.get("s3_bucket") and result.get("s3_key"):
                    f.write(f'S3 Location: s3://{result["s3_bucket"]}/{result["s3_key"]}\n')
                f.write(f'Recommended Dataset: {result.get("recommended_dataset", "N/A")}\n')
                f.write(f'"""\n\n')
                f.write(result["code"])
            
            # Save metadata
            metadata_file = filepath.replace('.py', '_metadata.json')
            metadata = {
                "paper_id": result.get("paper_id"),
                "paper_title": result.get("paper_title"),
                "paper_authors": result.get("paper_authors", []),
                "explanation": result.get("explanation"),
                "generated_at": result.get("generated_at"),
                "model_used": result.get("model_used"),
                "code_file": filepath,
                "s3_bucket": result.get("s3_bucket"),
                "s3_key": result.get("s3_key"),
                "recommended_dataset": result.get("recommended_dataset"),
                "dataset_recommendations": result.get("dataset_recommendations")
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved generated code to: {filepath}")
            logger.info(f"Saved metadata to: {metadata_file}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving generated code: {e}")
            return None


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Generate PyTorch code from papers in S3 (local testing only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--s3-bucket', help='S3 bucket name')
    input_group.add_argument('--paper-id', help='OpenSearch paper ID (fetches metadata from OpenSearch)')
    
    parser.add_argument('--s3-key', help='S3 key (required if using --s3-bucket)')
    parser.add_argument('--include-full-content', action='store_true', 
                       help='Include full paper content in generation')
    parser.add_argument('--output-dir', default='generated_code', 
                       help='Output directory for generated code (default: generated_code)')
    parser.add_argument('--clear-dir', action='store_true',
                       help='Clear output directory before saving')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save code to file, just print to stdout')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.s3_bucket and not args.s3_key:
        parser.error("--s3-key is required when using --s3-bucket")
    
    # Initialize generator
    generator = LocalCodeGenerator()
    
    # Generate code
    if args.paper_id:
        logger.info(f"Generating code from paper ID: {args.paper_id}")
        result = generator.generate_from_paper_id(
            args.paper_id,
            include_full_content=args.include_full_content
        )
    else:
        logger.info(f"Generating code from S3: s3://{args.s3_bucket}/{args.s3_key}")
        result = generator.generate_from_s3(
            args.s3_bucket,
            args.s3_key,
            include_full_content=args.include_full_content
        )
    
    # Print result
    if result.get('success'):
        print("\n" + "="*80)
        print("CODE GENERATION SUCCESSFUL")
        print("="*80)
        print(f"Paper: {result.get('paper_title', 'Unknown')}")
        print(f"Recommended Dataset: {result.get('recommended_dataset', 'N/A')}")
        print(f"Model Used: {result.get('model_used', 'N/A')}")
        print(f"Generated At: {result.get('generated_at', 'N/A')}")
        print(f"Code Length: {len(result.get('code', ''))} characters")
        print("="*80 + "\n")
        
        if not args.no_save:
            # Save to file
            saved_path = generator.save_code_locally(result, args.output_dir, clear_dir=args.clear_dir)
            if saved_path:
                print(f"✓ Code saved to: {saved_path}")
            else:
                print("✗ Failed to save code")
        else:
            # Print code to stdout
            print("\nGenerated Code:")
            print("-"*80)
            print(result.get('code', ''))
            print("-"*80)
        
        # Print explanation if available
        if result.get('explanation'):
            print("\nExplanation:")
            print("-"*80)
            print(result.get('explanation')[:500] + "..." if len(result.get('explanation', '')) > 500 else result.get('explanation'))
            print("-"*80)
    else:
        print("\n" + "="*80)
        print("CODE GENERATION FAILED")
        print("="*80)
        print(f"Error: {result.get('error', 'Unknown error')}")
        print("="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

