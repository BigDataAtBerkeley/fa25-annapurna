#!/usr/bin/env python3
"""
Test script to run the new PDF-based code generation approach on a single paper.
Shows which pages were selected, the reasoning, and the generated PyTorch code.
"""

import sys
import os
import json
from pathlib import Path

# Add code_gen to path
code_gen_path = Path(__file__).parent / "code_gen"
sys.path.insert(0, str(code_gen_path))

from chunked_generator import ChunkedPyTorchGenerator
from opensearch_client import OpenSearchClient
from pdf_processor import PDFProcessor
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_pages_with_classifier(pdf_bytes: bytes, pdf_processor: PDFProcessor):
    """
    Analyze all pages in the PDF and show which ones were selected by the classifier.
    
    Args:
        pdf_bytes: PDF file as bytes
        pdf_processor: PDFProcessor instance with classifier
    """
    import fitz
    
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_document)
    
    print(f"\n{'='*80}")
    print(f"PAGE RELEVANCE ANALYSIS (Total pages: {total_pages})")
    print(f"{'='*80}\n")
    
    page_analyses = []
    
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        text = page.get_text()
        
        # Get classifier prediction if available
        classifier_result = None
        if pdf_processor.classifier and pdf_processor.classifier.is_trained:
            try:
                classifier_result = pdf_processor.classifier.predict(text)
            except Exception as e:
                logger.warning(f"Classifier prediction failed for page {page_num + 1}: {e}")
        
        # Get heuristic analysis as fallback
        heuristic_result = pdf_processor.analyze_page_relevance(pdf_bytes, page_num)
        
        # Determine which method was used
        if classifier_result:
            is_relevant = classifier_result['is_relevant']
            confidence = classifier_result['confidence']
            method = "classifier"
        else:
            is_relevant = heuristic_result.get('is_relevant', False)
            confidence = heuristic_result.get('score', 0.0) / 100.0  # Normalize score
            method = "heuristic"
        
        # Get first few lines of text for preview
        text_preview = text[:200].replace('\n', ' ').strip()
        if len(text) > 200:
            text_preview += "..."
        
        page_analyses.append({
            'page_num': page_num + 1,
            'is_relevant': is_relevant,
            'confidence': confidence,
            'method': method,
            'text_preview': text_preview,
            'features': heuristic_result.get('features', {})
        })
    
    pdf_document.close()
    
    # Print analysis
    relevant_pages = [p for p in page_analyses if p['is_relevant']]
    irrelevant_pages = [p for p in page_analyses if not p['is_relevant']]
    
    print(f"✅ RELEVANT PAGES ({len(relevant_pages)}/{total_pages}):")
    print(f"{'-'*80}")
    for page in relevant_pages:
        print(f"Page {page['page_num']:3d} | Method: {page['method']:10s} | "
              f"Confidence: {page['confidence']:.3f} | {page['text_preview']}")
    
    print(f"\n❌ IRRELEVANT PAGES ({len(irrelevant_pages)}/{total_pages}):")
    print(f"{'-'*80}")
    for page in irrelevant_pages[:10]:  # Show first 10 irrelevant pages
        print(f"Page {page['page_num']:3d} | Method: {page['method']:10s} | "
              f"Confidence: {page['confidence']:.3f} | {page['text_preview']}")
    if len(irrelevant_pages) > 10:
        print(f"... and {len(irrelevant_pages) - 10} more irrelevant pages")
    
    return page_analyses


def show_chunk_summaries(chunk_results):
    """
    Display summaries for each chunk.
    
    Args:
        chunk_results: List of chunk result dictionaries
    """
    print(f"\n{'='*80}")
    print(f"CHUNK SUMMARIES ({len(chunk_results)} chunks)")
    print(f"{'='*80}\n")
    
    for i, chunk in enumerate(chunk_results, 1):
        if chunk.get('success'):
            print(f"Chunk {i}:")
            print(f"  Pages: {chunk.get('pages', 'N/A')}")
            print(f"  Summary length: {chunk.get('summary_length', 0):,} chars")
            print(f"  Processing time: {chunk.get('processing_time', 0):.2f}s")
            print(f"  Summary preview:")
            summary = chunk.get('summary', '')
            preview = summary[:300].replace('\n', ' ')
            if len(summary) > 300:
                preview += "..."
            print(f"    {preview}")
            print()
        else:
            print(f"Chunk {i}: ❌ FAILED - {chunk.get('error', 'Unknown error')}\n")


def main():
    """Main function to test code generation on a single paper."""
    
    if len(sys.argv) < 2:
        print("Usage: python test_single_paper.py <paper_id>")
        print("\nExample:")
        print("  python test_single_paper.py 9ad32df4723703816a30c7c6995b646ed4779f0bbad499182e94a56758637d7d")
        sys.exit(1)
    
    paper_id = sys.argv[1]
    
    print(f"{'='*80}")
    print(f"TESTING CODE GENERATION FOR PAPER: {paper_id}")
    print(f"{'='*80}\n")
    
    try:
        # Initialize components
        print("Initializing components...")
        opensearch_client = OpenSearchClient()
        pdf_processor = PDFProcessor(use_classifier=True)
        generator = ChunkedPyTorchGenerator(
            batch_size=8,
            pages_per_pdf_chunk=2,
            use_smart_pdf_chunking=True,
            max_pdf_chunks=15
        )
        print("✅ Components initialized\n")
        
        # Get paper
        print(f"Fetching paper {paper_id}...")
        paper = opensearch_client.get_paper_by_id(paper_id)
        if not paper:
            print(f"❌ Paper {paper_id} not found in OpenSearch")
            sys.exit(1)
        
        paper_title = paper.get('title', 'Unknown')
        print(f"✅ Found paper: {paper_title}\n")
        
        # Get PDF bytes
        pdf_bytes = opensearch_client.get_paper_pdf_bytes(paper)
        if not pdf_bytes:
            print(f"❌ Failed to retrieve PDF for paper {paper_id}")
            sys.exit(1)
        
        print(f"✅ Retrieved PDF ({len(pdf_bytes):,} bytes)\n")
        
        # Analyze pages with classifier
        page_analyses = analyze_pages_with_classifier(pdf_bytes, pdf_processor)
        
        # Identify relevant pages (this is what the generator will use)
        print(f"\n{'='*80}")
        print("IDENTIFYING RELEVANT PAGES FOR CHUNKING")
        print(f"{'='*80}\n")
        
        relevant_pages = pdf_processor.identify_relevant_pages(pdf_bytes, max_pages=30)
        print(f"Selected {len(relevant_pages)} relevant pages: {[p+1 for p in relevant_pages]}")
        
        # Get PDF chunks
        pdf_chunks = pdf_processor.split_pdf_into_chunks(
            pdf_bytes,
            pages_per_chunk=2,
            use_smart_chunking=True,
            max_chunks=15
        )
        print(f"\nPDF split into {len(pdf_chunks)} chunks:")
        for i, (start, end) in enumerate(pdf_chunks, 1):
            print(f"  Chunk {i}: pages {start+1}-{end}")
        
        # Generate code
        print(f"\n{'='*80}")
        print("GENERATING CODE")
        print(f"{'='*80}\n")
        
        result = generator.generate_code_for_paper(paper_id)
        
        if not result.get('success'):
            print(f"❌ Code generation failed: {result.get('error')}")
            sys.exit(1)
        
        # Show chunk summaries
        chunk_results = result.get('chunk_results', [])
        show_chunk_summaries(chunk_results)
        
        # Show final code
        print(f"{'='*80}")
        print("GENERATED PYTORCH CODE")
        print(f"{'='*80}\n")
        
        code = result.get('code', '')
        print(code)
        
        # Save code to file
        output_file = f"generated_code_{paper_id[:16]}.py"
        with open(output_file, 'w') as f:
            f.write(code)
        print(f"\n✅ Code saved to: {output_file}")
        
        # Show summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}\n")
        print(f"Paper ID: {paper_id}")
        print(f"Paper Title: {paper_title}")
        print(f"Total pages analyzed: {len(page_analyses)}")
        print(f"Relevant pages selected: {len(relevant_pages)}")
        print(f"Chunks processed: {len(pdf_chunks)}")
        print(f"Successful chunks: {sum(1 for c in chunk_results if c.get('success'))}")
        print(f"Code length: {len(code):,} characters")
        print(f"Total generation time: {result.get('total_generation_time', 0):.2f}s")
        print(f"Dataset recommended: {result.get('recommended_dataset', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

