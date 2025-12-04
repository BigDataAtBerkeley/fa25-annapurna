import boto3
import base64
import fitz  # PyMuPDF
import json
import argparse
import random
import os
from io import BytesIO
from dotenv import load_dotenv
from code_gen.opensearch_client import OpenSearchClient

load_dotenv()

# ---------------------------
# Bedrock Client Setup
# ---------------------------
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
prompt = """Your task is to analyze the provided PDF pages and provide a DETAILED SUMMARY that will be used to generate PyTorch code. Focus on extracting:

1. MATHEMATICAL FORMULAS and equations - transcribe them exactly as they appear, including all notation
2. DIAGRAMS and figures - describe the architecture, data flow, network structures, and visual elements in detail
3. KEY ALGORITHMS AND METHODS described in these pages
4. ARCHITECTURAL DETAILS (neural network structures, layer types, connections, etc.)
5. TRAINING PROCEDURES (loss functions, optimization methods, training steps)
6. KEY IMPLEMENTATION DETAILS (data preprocessing, specific operations, etc.)
7. IMPORTANT CONSTANTS, HYPERPARAMETERS, or configuration details
8. DATASET INFORMATION (dataset names, data types, task types mentioned)
9. ANY CODE-RELEVANT INFORMATION that would be needed to implement this

CRITICAL: Pay special attention to formulas and diagrams that may not be fully captured in text. 
- For formulas: Write them out in LaTeX notation or clear mathematical notation
- For diagrams: Describe the structure, connections, and flow in detail
- For tables: Extract all numerical values and relationships
- For datasets: Note any dataset names, data types (images, text, etc.), or task types (classification, regression, etc.)

PRIORITY: Focus on content that can be directly implemented in code. If the pages contain only:
- Theoretical proofs without implementation details → Note this but focus on any implementable aspects
- Examples or qualitative discussions → Extract any concrete details that could inform implementation
- Evaluation results → Note key metrics but prioritize implementation details

Be extremely detailed and specific. Include all mathematical notation, formulas, and technical details.
This summary will be combined with summaries from other chunks to generate complete PyTorch code.

Format your response as a structured summary with clear sections.
"""


# ---------------------------
# Utility — extract single page from PDF bytes
# ---------------------------
def extract_single_page_pdf(pdf_bytes: bytes, page_num: int) -> bytes:
    """Extract a single page from PDF and return as PDF bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
    
    # Create new PDF with just this page
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    
    # Convert to bytes
    pdf_bytes_out = new_doc.tobytes()
    doc.close()
    new_doc.close()
    
    return pdf_bytes_out

# ---------------------------
# 1. TEXT-ONLY MODE (PyMuPDF text extraction)
# ---------------------------
def analyze_pdf_text_only(pdf_bytes: bytes):
    # Extract text from the specific page using PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    page = doc.load_page(0)
    extracted_text = page.get_text()
    doc.close()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\n\nExtracted text from PDF page:\n\n{extracted_text}"
                    }
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    output = json.loads(response["body"].read())
    return output["content"][0]["text"]


# ---------------------------
# 2. DIRECT PDF → CLAUDE VISUAL MODE (VISION)
# ---------------------------
def analyze_pdf_visual(pdf_bytes: bytes):
    """
    Send PDF directly to Claude using the document format.
    Ensures proper PDF structure and base64 encoding.
    """
    # Verify PDF is valid and properly formatted
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Re-save to ensure clean PDF structure
        clean_pdf_bytes = doc.tobytes(garbage=4, deflate=True)
        doc.close()
    except Exception as e:
        print(f"Warning: Could not clean PDF, using original: {e}")
        clean_pdf_bytes = pdf_bytes
    
    pdf_b64 = base64.b64encode(clean_pdf_bytes).decode("utf-8")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0
    }

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body)
        )
        output = json.loads(response["body"].read())
        return output["content"][0]["text"]
    except Exception as e:
        print(f"Bedrock API Error: {e}")
        # Try to print the response if available
        if hasattr(e, 'response'):
            print(f"Response: {e.response}")
        raise
# ---------------------------
# 3. PYMU CONVERSION → IMAGES → CLAUDE VISION
# ---------------------------
def analyze_pdf_as_images(pdf_bytes: bytes, page_num: int):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} out of range (PDF has {len(doc)} pages)")
    
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    doc.close()

    content_blocks = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64}
        },
        {
            "type": "text",
            "text": (prompt)
        }
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": content_blocks
            }
        ],
        "max_tokens": 4000,
        "temperature": 0
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    output = json.loads(response["body"].read())
    return output["content"][0]["text"]


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Bedrock PDF extraction methods")
    parser.add_argument("--paper-id", required=True, help="Paper ID from OpenSearch")
    parser.add_argument("--page-number", type=int, help="Page number (0-indexed). If not provided, randomly sampled.")
    
    args = parser.parse_args()
    
    # Get paper from OpenSearch
    opensearch_client = OpenSearchClient()
    paper = opensearch_client.get_paper_by_id(args.paper_id)
    
    if not paper:
        print(f"Error: Paper {args.paper_id} not found in OpenSearch")
        exit(1)
    
    s3_bucket = paper.get('s3_bucket')
    s3_key = paper.get('s3_key')
    
    if not s3_bucket or not s3_key:
        print(f"Error: Paper {args.paper_id} missing S3 bucket or key")
        exit(1)
    
    # Download PDF from S3
    print(f"Downloading PDF from s3://{s3_bucket}/{s3_key}...")
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        pdf_bytes = response['Body'].read()
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        exit(1)
    
    # Get page number
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    doc.close()
    
    if args.page_number is not None:
        page_num = args.page_number
        if page_num < 0 or page_num >= total_pages:
            print(f"Error: Page {page_num} out of range (PDF has {total_pages} pages, 0-indexed)")
            exit(1)
    else:
        page_num = random.randint(0, total_pages - 1)
        print(f"Randomly selected page {page_num} (out of {total_pages} pages)")
    
    # Extract single page PDF
    single_page_pdf = extract_single_page_pdf(pdf_bytes, page_num)
    
    print(f"\nProcessing page {page_num} of paper {args.paper_id} ({s3_key})")
    print("=" * 70)
    
    # Method 1: Text-only mode
    print("\n======== TEXT-ONLY MODE ========\n")
    try:
        result_text = analyze_pdf_text_only(single_page_pdf)
        print(result_text)
    except Exception as e:
        print(f"Error in text-only mode: {e}")
    
    # Method 2: Visual mode
    print("\n======== VISUAL MODE ========\n")
    try:
        result_visual = analyze_pdf_visual(single_page_pdf)
        print(result_visual)
    except Exception as e:
        print(f"Error in visual mode: {e}")
    
    # Method 3: Image-based mode
    print("\n======== IMAGE-BASED MODE (PyMu → Images) ========\n")
    try:
        result_images = analyze_pdf_as_images(pdf_bytes, page_num)
        print(result_images)
    except Exception as e:
        print(f"Error in image-based mode: {e}")
