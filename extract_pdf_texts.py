"""
Script to extract text from PDF pages for randomly sampled papers.
Creates a CSV with paper_id, pdf_name, page_number, and extracted text.
"""

import os
import csv
import boto3
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")

from code_gen.opensearch_client import OpenSearchClient


def extract_text_from_pdf(pdf_bytes: bytes) -> List[str]:
    """Extract text from each page of a PDF."""
    texts = []
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            texts.append(text)
        pdf_document.close()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return texts


def main():
    """Main function to sample papers, extract PDF text, and write to CSV."""
    # Initialize clients
    opensearch_client = OpenSearchClient()
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Get 50 random papers
    print("Fetching 50 random papers from OpenSearch...")
    papers = opensearch_client.get_random_papers(size=50)
    print(f"Retrieved {len(papers)} papers")
    
    # Prepare CSV data
    csv_rows = []
    
    for paper in papers:
        paper_id = paper.get('_id') or paper.get('id', 'unknown')
        s3_bucket = paper.get('s3_bucket')
        s3_key = paper.get('s3_key')
        
        if not s3_bucket or not s3_key:
            print(f"Skipping paper {paper_id}: missing S3 bucket or key")
            continue
        
        if not s3_key.lower().endswith('.pdf'):
            print(f"Skipping paper {paper_id}: not a PDF file")
            continue
        
        print(f"Processing paper {paper_id}: {s3_key}")
        
        # Download PDF from S3
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            pdf_bytes = response['Body'].read()
        except Exception as e:
            print(f"Error downloading PDF for {paper_id}: {e}")
            continue
        
        # Extract text from each page
        page_texts = extract_text_from_pdf(pdf_bytes)
        
        # Add rows for each page
        for page_num, text in enumerate(page_texts, start=1):
            csv_rows.append({
                'paper_id': paper_id,
                'pdf_name': s3_key,
                'page_number': page_num,
                'text': text
            })
        
        print(f"Extracted text from {len(page_texts)} pages for {paper_id}")
    
    # Write to CSV
    output_file = 'pdf_texts.csv'
    print(f"Writing {len(csv_rows)} rows to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['paper_id', 'pdf_name', 'page_number', 'text'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"✅ Completed! Wrote {len(csv_rows)} rows to {output_file}")


if __name__ == '__main__':
    main()

