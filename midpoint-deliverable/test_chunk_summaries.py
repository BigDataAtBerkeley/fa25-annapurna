#!/usr/bin/env python3
"""
Test script to view chunk summaries for a paper.

Usage:
    # Activate virtual environment first (if using aws_env):
    # source ../aws_env/bin/activate
    
    python test_chunk_summaries.py --paper-id <paper_id>
    python test_chunk_summaries.py --s3-bucket <bucket> --s3-key <key>

Note: If using --paper-id, you need opensearch-py installed:
    pip install opensearch-py
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add code-gen-for-deliv to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code-gen-for-deliv'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code-gen-for-deliv', 'chunked-code-gen'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules (with graceful handling)
OpenSearchClient = None
try:
    from opensearch_client import OpenSearchClient
    OPENSEARCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenSearch client not available: {e}")
    logger.warning("OpenSearch functionality will be disabled (use --s3-bucket/--s3-key instead)")
    OPENSEARCH_AVAILABLE = False

try:
    from pdf_processor import PDFProcessor
    PDF_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"PDF processor not available: {e}")
    logger.error("PDF processing is required. Install dependencies: pip install pymupdf Pillow")
    sys.exit(1)

try:
    from chunked_bedrock_client import ChunkedBedrockClient
    BEDROCK_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Bedrock client not available: {e}")
    logger.error("Bedrock client is required. Check your imports.")
    sys.exit(1)


def get_pdf_from_opensearch(paper_id: str) -> tuple:
    """Get PDF bytes and paper summary from OpenSearch."""
    if not OPENSEARCH_AVAILABLE or not OpenSearchClient:
        logger.error("OpenSearch client is not available")
        logger.error("Install opensearch-py: pip install opensearch-py")
        logger.error("Or use --s3-bucket/--s3-key instead")
        return None, None, None
    
    try:
        opensearch_client = OpenSearchClient()
    except Exception as e:
        logger.error(f"Failed to initialize OpenSearch client: {e}")
        logger.error("Make sure OPENSEARCH_ENDPOINT and AWS credentials are set")
        return None, None, None
    
    # Get paper from OpenSearch
    paper = opensearch_client.get_paper_by_id(paper_id)
    if not paper:
        logger.error(f"Paper {paper_id} not found in OpenSearch")
        return None, None, None
    
    paper_summary = opensearch_client.get_paper_summary(paper)
    
    # Check if it's a PDF
    if not opensearch_client.is_pdf_paper(paper):
        logger.error(f"Paper {paper_id} is not a PDF")
        return None, None, None
    
    # Get PDF bytes
    pdf_bytes = opensearch_client.get_paper_pdf_bytes(paper)
    if not pdf_bytes:
        logger.error(f"Failed to retrieve PDF for paper {paper_id}")
        return None, None, None
    
    logger.info(f"✅ Retrieved PDF for paper: {paper.get('title', 'Unknown')}")
    logger.info(f"   PDF size: {len(pdf_bytes):,} bytes")
    
    return pdf_bytes, paper_summary, paper


def get_pdf_from_s3(s3_bucket: str, s3_key: str) -> tuple:
    """Get PDF bytes directly from S3."""
    pdf_processor = PDFProcessor()
    pdf_bytes = pdf_processor.download_pdf_from_s3(s3_bucket, s3_key)
    
    if not pdf_bytes:
        logger.error(f"Failed to download PDF from s3://{s3_bucket}/{s3_key}")
        return None, None, None
    
    logger.info(f"✅ Downloaded PDF from s3://{s3_bucket}/{s3_key}")
    logger.info(f"   PDF size: {len(pdf_bytes):,} bytes")
    
    # Create a minimal paper summary
    paper_summary = f"Paper from s3://{s3_bucket}/{s3_key}"
    
    return pdf_bytes, paper_summary, None


def process_pdf_chunks(pdf_bytes: bytes, paper_summary: str, 
                      use_smart_chunking: bool = True, 
                      max_chunks: int = 15,
                      pages_per_chunk: int = 2) -> list:
    """Process PDF chunks and return summaries."""
    pdf_processor = PDFProcessor()
    bedrock_client = ChunkedBedrockClient()
    
    # Split PDF into chunks
    logger.info(f"📄 Splitting PDF into chunks (smart chunking: {use_smart_chunking})...")
    pdf_chunks = pdf_processor.split_pdf_into_chunks(
        pdf_bytes,
        pages_per_chunk=pages_per_chunk,
        use_smart_chunking=use_smart_chunking,
        max_chunks=max_chunks
    )
    
    if not pdf_chunks:
        logger.error("Failed to split PDF into chunks")
        return []
    
    logger.info(f"✅ Split PDF into {len(pdf_chunks)} chunks")
    
    # Process each chunk
    chunk_summaries = []
    for i, (page_start, page_end) in enumerate(pdf_chunks, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing chunk {i}/{len(pdf_chunks)} (pages {page_start + 1}-{page_end})...")
        
        # Extract pages as images
        base64_images = pdf_processor.process_pdf_chunk(
            pdf_bytes, page_start, page_end, dpi=150
        )
        
        if not base64_images:
            logger.warning(f"⚠️  Failed to extract images for chunk {i}")
            chunk_summaries.append({
                "chunk_number": i,
                "pages": f"{page_start + 1}-{page_end}",
                "summary": None,
                "error": "Failed to extract images"
            })
            continue
        
        # Generate summary using vision
        result = bedrock_client.summarize_pdf_chunk_with_vision(
            base64_images=base64_images,
            chunk_number=i,
            total_chunks=len(pdf_chunks),
            paper_summary=paper_summary,
            page_start=page_start,
            page_end=page_end
        )
        
        if result.get("success"):
            summary = result["summary"]
            chunk_summaries.append({
                "chunk_number": i,
                "pages": f"{page_start + 1}-{page_end}",
                "summary": summary,
                "summary_length": len(summary),
                "num_images": len(base64_images)
            })
            logger.info(f"✅ Chunk {i} summarized ({len(summary):,} chars)")
        else:
            logger.error(f"❌ Chunk {i} failed: {result.get('error', 'Unknown error')}")
            chunk_summaries.append({
                "chunk_number": i,
                "pages": f"{page_start + 1}-{page_end}",
                "summary": None,
                "error": result.get("error", "Unknown error")
            })
    
    return chunk_summaries


def print_chunk_summaries(chunk_summaries: list, output_file: str = None):
    """Print chunk summaries in a readable format."""
    print("\n" + "="*80)
    print("CHUNK SUMMARIES")
    print("="*80 + "\n")
    
    for chunk in chunk_summaries:
        print(f"{'='*80}")
        print(f"CHUNK {chunk['chunk_number']} - Pages {chunk['pages']}")
        print(f"{'='*80}")
        
        if chunk.get('error'):
            print(f"❌ ERROR: {chunk['error']}\n")
            continue
        
        if chunk.get('summary'):
            print(f"Summary length: {chunk.get('summary_length', 0):,} characters")
            print(f"Images processed: {chunk.get('num_images', 0)}")
            print("\n" + "-"*80)
            print("SUMMARY:")
            print("-"*80)
            print(chunk['summary'])
            print("\n")
        else:
            print("⚠️  No summary available\n")
    
    # Save to file if requested
    if output_file:
        import json
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_summaries, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved chunk summaries to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test script to view chunk summaries for a paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From OpenSearch paper ID
  python test_chunk_summaries.py --paper-id C-hIZpoBclM7MZc3XpR7
  
  # From S3 directly
  python test_chunk_summaries.py --s3-bucket my-bucket --s3-key papers/paper.pdf
  
  # Save to file
  python test_chunk_summaries.py --paper-id C-hIZpoBclM7MZc3XpR7 --output summaries.json
  
  # Disable smart chunking (process all pages)
  python test_chunk_summaries.py --paper-id C-hIZpoBclM7MZc3XpR7 --no-smart-chunking
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--paper-id', type=str, 
                            help='OpenSearch paper ID (requires opensearch-py)')
    input_group.add_argument('--s3-bucket', type=str, help='S3 bucket name')
    
    parser.add_argument('--s3-key', type=str, help='S3 key (required if --s3-bucket is used)')
    parser.add_argument('--output', type=str, help='Output file to save summaries (JSON format)')
    parser.add_argument('--no-smart-chunking', action='store_true', 
                       help='Disable smart chunking (process all pages)')
    parser.add_argument('--max-chunks', type=int, default=15,
                       help='Maximum number of chunks to process (default: 15)')
    parser.add_argument('--pages-per-chunk', type=int, default=2,
                       help='Number of pages per chunk (default: 2)')
    
    args = parser.parse_args()
    
    # Validate S3 arguments
    if args.s3_bucket and not args.s3_key:
        parser.error("--s3-key is required when using --s3-bucket")
    
    # Check if OpenSearch is needed
    if args.paper_id and not OPENSEARCH_AVAILABLE:
        logger.error("❌ --paper-id requires OpenSearch client")
        logger.error("Install opensearch-py: pip install opensearch-py")
        logger.error("Or use --s3-bucket/--s3-key instead")
        sys.exit(1)
    
    # Get PDF bytes
    if args.paper_id:
        pdf_bytes, paper_summary, paper = get_pdf_from_opensearch(args.paper_id)
        if not pdf_bytes:
            logger.error("Failed to retrieve PDF from OpenSearch")
            sys.exit(1)
    else:
        pdf_bytes, paper_summary, paper = get_pdf_from_s3(args.s3_bucket, args.s3_key)
        if not pdf_bytes:
            logger.error("Failed to retrieve PDF from S3")
            sys.exit(1)
    
    # Process chunks
    use_smart_chunking = not args.no_smart_chunking
    chunk_summaries = process_pdf_chunks(
        pdf_bytes,
        paper_summary,
        use_smart_chunking=use_smart_chunking,
        max_chunks=args.max_chunks,
        pages_per_chunk=args.pages_per_chunk
    )
    
    if not chunk_summaries:
        logger.error("No chunk summaries generated")
        sys.exit(1)
    
    # Print summaries
    print_chunk_summaries(chunk_summaries, args.output)
    
    # Summary stats
    successful = sum(1 for c in chunk_summaries if c.get('summary'))
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total chunks: {len(chunk_summaries)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(chunk_summaries) - successful}")
    if successful > 0:
        total_chars = sum(c.get('summary_length', 0) for c in chunk_summaries if c.get('summary'))
        avg_chars = total_chars / successful
        print(f"Total summary length: {total_chars:,} characters")
        print(f"Average summary length: {avg_chars:,.0f} characters")
    print()


if __name__ == "__main__":
    main()

