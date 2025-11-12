"""
PDF processing utilities for extracting page chunks and converting to images.
Supports extracting formulas and diagrams from PDF pages for Bedrock vision analysis.
"""

import os
import io
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple
import boto3
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Install with: pip install pymupdf")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available. Install with: pip install Pillow")


class PDFProcessor:
    """Process PDF files to extract page chunks and convert to images for vision analysis."""
    
    def __init__(self, aws_region: str = "us-east-1"):
        """
        Initialize PDF processor.
        
        Args:
            aws_region: AWS region for S3 access
        """
        self.aws_region = aws_region
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install pymupdf")
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required for image processing. Install with: pip install Pillow")
        
        logger.info("PDF processor initialized")
    
    def download_pdf_from_s3(self, s3_bucket: str, s3_key: str) -> Optional[bytes]:
        """
        Download PDF file from S3.
        
        Args:
            s3_bucket: S3 bucket name
            s3_key: S3 object key
            
        Returns:
            PDF file bytes or None if download fails
        """
        try:
            response = self.s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            pdf_bytes = response['Body'].read()
            logger.info(f"Downloaded PDF from s3://{s3_bucket}/{s3_key} ({len(pdf_bytes)} bytes)")
            return pdf_bytes
        except Exception as e:
            logger.error(f"Error downloading PDF from S3: {e}")
            return None
    
    def is_pdf_file(self, s3_key: str) -> bool:
        """
        Check if S3 key points to a PDF file.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if file appears to be a PDF
        """
        return s3_key.lower().endswith('.pdf')
    
    def get_pdf_page_count(self, pdf_bytes: bytes) -> int:
        """
        Get the number of pages in a PDF.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Number of pages
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_count = len(pdf_document)
            pdf_document.close()
            return page_count
        except Exception as e:
            logger.error(f"Error getting PDF page count: {e}")
            return 0
    
    def extract_pdf_pages_as_images(self, pdf_bytes: bytes, page_start: int, page_end: int, 
                                     dpi: int = 150) -> List[bytes]:
        """
        Extract a range of PDF pages and convert them to PNG images.
        
        Args:
            pdf_bytes: PDF file as bytes
            page_start: Starting page number (0-indexed)
            page_end: Ending page number (exclusive, 0-indexed)
            dpi: Resolution for rendering (default: 150)
            
        Returns:
            List of PNG image bytes (one per page)
        """
        images = []
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            
            # Clamp page range to valid pages
            page_start = max(0, min(page_start, total_pages - 1))
            page_end = max(page_start + 1, min(page_end, total_pages))
            
            for page_num in range(page_start, page_end):
                page = pdf_document[page_num]
                
                # Render page to image (pixmap)
                # Use zoom factor to control DPI: zoom = dpi / 72 (default PDF DPI)
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PNG bytes
                img_bytes = pix.tobytes("png")
                images.append(img_bytes)
                
                logger.debug(f"Extracted page {page_num + 1} as image ({len(img_bytes)} bytes)")
            
            pdf_document.close()
            logger.info(f"Extracted {len(images)} pages ({page_start + 1}-{page_end}) as images")
            return images
            
        except Exception as e:
            logger.error(f"Error extracting PDF pages as images: {e}")
            return []
    
    def images_to_base64(self, image_bytes_list: List[bytes]) -> List[str]:
        """
        Convert image bytes to base64-encoded strings for Bedrock API.
        
        Args:
            image_bytes_list: List of image bytes (PNG format)
            
        Returns:
            List of base64-encoded image strings
        """
        base64_images = []
        for img_bytes in image_bytes_list:
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(base64_str)
        return base64_images
    
    def split_pdf_into_chunks(self, pdf_bytes: bytes, pages_per_chunk: int = 2) -> List[Tuple[int, int]]:
        """
        Split PDF into page ranges (chunks).
        
        Args:
            pdf_bytes: PDF file as bytes
            pages_per_chunk: Number of pages per chunk (default: 2)
            
        Returns:
            List of (start_page, end_page) tuples (0-indexed, end_page is exclusive)
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            pdf_document.close()
            
            chunks = []
            for start in range(0, total_pages, pages_per_chunk):
                end = min(start + pages_per_chunk, total_pages)
                chunks.append((start, end))
            
            logger.info(f"Split PDF into {len(chunks)} chunks ({pages_per_chunk} pages per chunk, {total_pages} total pages)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting PDF into chunks: {e}")
            return []
    
    def process_pdf_chunk(self, pdf_bytes: bytes, page_start: int, page_end: int, 
                         dpi: int = 150) -> Optional[List[str]]:
        """
        Process a PDF chunk: extract pages and convert to base64-encoded images.
        
        Args:
            pdf_bytes: PDF file as bytes
            page_start: Starting page number (0-indexed)
            page_end: Ending page number (exclusive, 0-indexed)
            dpi: Resolution for rendering (default: 150)
            
        Returns:
            List of base64-encoded image strings, or None if processing fails
        """
        images = self.extract_pdf_pages_as_images(pdf_bytes, page_start, page_end, dpi)
        if not images:
            return None
        
        base64_images = self.images_to_base64(images)
        return base64_images

