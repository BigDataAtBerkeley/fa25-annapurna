"""
PDF processing utilities for extracting page chunks and converting to PDF bytes.
Supports extracting formulas and diagrams from PDF pages for Bedrock analysis.
Uses a trained classifier to determine which pages are relevant for code generation.
"""

import os
import io
import base64
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import boto3
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from page_classifier import PageRelevanceClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    logger.warning("Page classifier not available. Install required dependencies.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info(f"✅ PyMuPDF imported successfully (version: {getattr(fitz, 'version', 'unknown')})")
except ImportError as e:
    PYMUPDF_AVAILABLE = False
    logger.warning(f"PyMuPDF not available. ImportError: {e}")
except Exception as e:
    PYMUPDF_AVAILABLE = False
    logger.error(f"PyMuPDF import failed with exception: {type(e).__name__}: {e}")



class PDFProcessor:
    """Process PDF files to extract page chunks and convert to PDF bytes for analysis."""
    
    def __init__(self, aws_region: str = "us-east-1", use_classifier: bool = True):
        """
        Initialize PDF processor.
        
        Args:
            aws_region: AWS region for S3 access
            use_classifier: Whether to use the page relevance classifier (default: True)
        """
        self.aws_region = aws_region
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.use_classifier = use_classifier
        
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing. Install with: pip install pymupdf")
        
        # Lazy-load classifier to avoid hanging during initialization
        self._classifier = None
        self.use_classifier = use_classifier
        
        logger.info("PDF processor initialized (classifier will be loaded on first use)")
    
    @property
    def classifier(self):
        """Lazy-load classifier only when needed."""
        if self._classifier is None and self.use_classifier:
            try:
                logger.info("Loading page relevance classifier (lazy load)...")
                self._classifier = PageRelevanceClassifier()
                logger.info("Classifier loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                self._classifier = None
        return self._classifier
    
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
    
    def analyze_page_relevance(self, pdf_bytes: bytes, page_num: int) -> Dict[str, Any]:
        """
        Analyze a single page to determine its relevance for code generation.
        Uses trained classifier if available, otherwise falls back to heuristic scoring.
        
        Args:
            pdf_bytes: PDF file as bytes
            page_num: Page number (0-indexed)
            
        Returns:
            Dictionary with relevance score and detected features
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            if page_num >= len(pdf_document):
                pdf_document.close()
                return {"score": 0, "features": {}, "is_relevant": False}
            
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Use classifier if available
            if self.classifier and self.classifier.is_trained:
                try:
                    prediction = self.classifier.predict(text)
                    is_relevant = prediction['is_relevant']
                    confidence = prediction['confidence']
                    
                    # Convert to score (0-100 scale for compatibility)
                    score = confidence * 100.0 if is_relevant else (1 - confidence) * 100.0
                    
                    logger.info(f"Page {page_num + 1}: Classifier prediction - relevant={is_relevant}, confidence={confidence:.3f}, score={score:.1f}")
                    
                    pdf_document.close()
                    return {
                        "score": score,
                        "is_relevant": is_relevant,
                        "confidence": confidence,
                        "features": {"classifier_used": True},
                        "method": "classifier"
                    }
                except Exception as e:
                    logger.warning(f"Classifier prediction failed for page {page_num}: {e}. Using fallback.")
            
            # Fallback to heuristic scoring
            text_lower = text.lower()
            
            # Extract images/drawings (diagrams, figures)
            image_list = page.get_images()
            drawing_list = page.get_drawings()
            has_images = len(image_list) > 0 or len(drawing_list) > 0
            
            # Score based on various factors
            score = 0.0
            features = {
                "has_abstract": False,
                "has_formulas": False,
                "has_diagrams": False,
                "has_key_terms": False,
                "has_algorithms": False,
                "has_architecture": False,
                "has_results": False,
                "section_type": None,
                "formula_density": 0.0,
                "image_count": len(image_list) + len(drawing_list),
                "classifier_used": False
            }
            
            # 1. Check for abstract (usually first 2-3 pages)
            abstract_keywords = ['abstract', 'summary']
            if page_num < 3 and any(kw in text_lower for kw in abstract_keywords):
                score += 10.0
                features["has_abstract"] = True
                features["section_type"] = "abstract"
            
            # 2. Detect mathematical formulas
            # Look for common math notation patterns
            math_patterns = [
                r'\\[\(\[].*?\\[\)\]]',  # LaTeX-style formulas
                r'∑|∏|∫|∂|∇|α|β|γ|θ|λ|μ|σ|π|∞',  # Math symbols
                r'[a-z]\s*=\s*[a-z]',  # Equations like "x = y"
                r'[a-z]\([a-z]\)',  # Functions like "f(x)"
                r'\^[0-9]|_[0-9]',  # Superscripts/subscripts
                r'\\frac|\\sqrt|\\sum|\\int',  # LaTeX commands
            ]
            
            formula_count = 0
            for pattern in math_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                formula_count += len(matches)
            
            if formula_count > 0:
                score += min(formula_count * 2.0, 15.0)  # Cap at 15 points
                features["has_formulas"] = True
                features["formula_density"] = formula_count / max(len(text.split()), 1)
            
            # 3. Detect diagrams/figures
            if has_images:
                score += 8.0
                features["has_diagrams"] = True
            
            # Check for figure captions
            figure_keywords = ['figure', 'fig.', 'diagram', 'architecture', 'network structure']
            if any(kw in text_lower for kw in figure_keywords):
                score += 5.0
                features["has_diagrams"] = True
            
            # 4. Key terms for ML/DL papers
            key_terms = [
                'neural network', 'convolutional', 'transformer', 'attention', 'loss function',
                'optimizer', 'gradient', 'backpropagation', 'activation', 'layer', 'model',
                'training', 'epoch', 'batch', 'learning rate', 'architecture', 'algorithm',
                'pytorch', 'tensorflow', 'implementation', 'code', 'pseudocode'
            ]
            
            term_count = sum(1 for term in key_terms if term in text_lower)
            if term_count > 0:
                score += min(term_count * 1.5, 10.0)  # Cap at 10 points
                features["has_key_terms"] = True
            
            # 5. Algorithm descriptions
            algorithm_keywords = ['algorithm', 'pseudocode', 'procedure', 'method', 'approach']
            if any(kw in text_lower for kw in algorithm_keywords):
                score += 6.0
                features["has_algorithms"] = True
            
            # 6. Architecture descriptions
            architecture_keywords = ['architecture', 'network structure', 'model structure', 'layer', 'neural network']
            if any(kw in text_lower for kw in architecture_keywords):
                score += 7.0
                features["has_architecture"] = True
            
            # 7. Section headers (important sections)
            section_headers = {
                'introduction': 3.0,
                'methodology': 8.0,
                'method': 8.0,
                'approach': 8.0,
                'architecture': 10.0,
                'model': 10.0,
                'implementation': 9.0,
                'experiments': 4.0,
                'results': 4.0,
                'discussion': 2.0,
                'conclusion': 2.0,
                'related work': 1.0,
                'references': 0.0
            }
            
            for section, section_score in section_headers.items():
                # Look for section headers (usually at start of line, possibly numbered)
                pattern = rf'^\s*\d*\.?\s*{section}'
                if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
                    score += section_score
                    if not features["section_type"]:
                        features["section_type"] = section
                    break
            
            # 8. Results/tables
            if 'table' in text_lower or 'result' in text_lower or 'accuracy' in text_lower or 'loss' in text_lower:
                score += 3.0
                features["has_results"] = True
            
            # 9. Boost score for early pages (abstract, intro usually more important)
            if page_num < 5:
                score += 2.0
            
            # Determine if relevant (threshold: score >= 10)
            is_relevant = score >= 10.0
            
            features["score"] = score
            pdf_document.close()
            
            return {
                "score": score,
                "is_relevant": is_relevant,
                "features": features,
                "method": "heuristic"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing page {page_num} relevance: {e}")
            return {"score": 0.0, "features": {}}
    
    def identify_relevant_pages(self, pdf_bytes: bytes, max_pages: int = 20) -> List[int]:
        """
        Identify the most relevant pages in a PDF for code generation.
        Analyzes all pages and returns the top N most relevant page numbers.
        
        Args:
            pdf_bytes: PDF file as bytes
            max_pages: Maximum number of relevant pages to return (default: 20)
            
        Returns:
            List of page numbers (0-indexed) sorted by relevance (most relevant first)
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            pdf_document.close()
            
            logger.info(f"🔍 Using trained classifier to analyze {total_pages} pages for relevance")
            
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_texts = []
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                page_texts.append(page.get_text())
            pdf_document.close()
            
            logger.info(f"Running batch classifier prediction on {len(page_texts)} pages...")
            predictions = self.classifier.predict_batch(page_texts)
            
            page_scores = []
            relevant_count = 0
            for page_num, pred in enumerate(predictions):
                is_relevant = pred["is_relevant"]
                confidence = pred["confidence"]
                score = confidence * 100.0 if is_relevant else (1 - confidence) * 100.0
                page_scores.append((page_num, score, is_relevant, {
                    "classifier_used": True,
                    "confidence": confidence,
                    "method": "classifier"
                }))
                if is_relevant:
                    relevant_count += 1
                    logger.info(f"Page {page_num + 1}: Classifier - relevant=True, confidence={confidence:.3f}, score={score:.1f}")
            
            logger.info(f"✅ Batch classifier analysis complete: {relevant_count}/{total_pages} pages marked as relevant")
            
            # Filter to only relevant pages, then sort by score
            relevant_pages_with_scores = [(p, s, f) for p, s, is_rel, f in page_scores if is_rel]
            
            # Only use pages marked as relevant (no fallback to score-based selection)
            if relevant_pages_with_scores:
                relevant_pages_with_scores.sort(key=lambda x: x[1], reverse=True)
                # Take up to max_pages of the relevant pages
                relevant_pages = [p for p, s, f in relevant_pages_with_scores[:max_pages]]
                logger.info(f"Using {len(relevant_pages)} pages that were marked as relevant (out of {len(relevant_pages_with_scores)} total relevant pages, max_pages={max_pages})")
            else:
                # No pages marked as relevant - fallback to first 2 pages (abstract/intro)
                logger.warning(f"⚠️ No pages marked as relevant by classifier, falling back to first 2 pages")
                relevant_pages = list(range(min(2, total_pages)))
            
            # Always include first 2 pages (abstract/intro) if not already included
            for page_num in range(min(2, total_pages)):
                if page_num not in relevant_pages:
                    relevant_pages.append(page_num)
                    logger.info(f"Added page {page_num + 1} (abstract/intro) to relevant pages")
            
            # Sort by page number to maintain order
            relevant_pages = sorted(set(relevant_pages))
            
            # Sort by score for display
            sorted_scores = sorted(page_scores, key=lambda x: x[1], reverse=True)
            
            logger.info(f"Identified {len(relevant_pages)} relevant pages out of {total_pages} total pages")
            if sorted_scores:
                top_10 = sorted_scores[:10]
                top_pages_str = ", ".join([f"p{p+1}({s:.1f})" for p, s, _, _ in top_10])
                logger.info(f"Top 10 pages by relevance score: {top_pages_str}")
                
                # Show which pages were selected
                selected_pages_str = ", ".join([f"p{p+1}" for p in sorted(relevant_pages)])
                logger.info(f"Selected pages for processing: {selected_pages_str}")
                
                # Show classifier vs heuristic breakdown for selected pages
                selected_methods = {}
                for p in relevant_pages:
                    _, _, _, features = page_scores[p]
                    method = features.get('method', 'heuristic')
                    selected_methods[method] = selected_methods.get(method, 0) + 1
                if selected_methods:
                    method_str = ", ".join([f"{method}: {count}" for method, count in selected_methods.items()])
                    logger.info(f"Selected pages by method: {method_str}")
            
            return relevant_pages
            
        except Exception as e:
            logger.error(f"Error identifying relevant pages: {e}")
            # Fallback: return first N pages
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            pdf_document.close()
            return list(range(min(max_pages, total_pages)))
    
    def split_pdf_into_chunks(self, pdf_bytes: bytes, pages_per_chunk: int = 2, 
                            use_smart_chunking: bool = True, max_chunks: int = 15) -> List[Tuple[int, int]]:
        """
        Split PDF into page ranges (chunks) using smart chunking that prioritizes relevant sections.
        
        Args:
            pdf_bytes: PDF file as bytes
            pages_per_chunk: Number of pages per chunk (default: 2)
            use_smart_chunking: If True, only process relevant pages (abstract, formulas, diagrams, etc.)
            max_chunks: Maximum number of chunks to return when using smart chunking (default: 15)
            
        Returns:
            List of (start_page, end_page) tuples (0-indexed, end_page is exclusive)
        """
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(pdf_document)
            pdf_document.close()
            
            if use_smart_chunking:
                # Use smart chunking: identify relevant pages first
                relevant_pages = self.identify_relevant_pages(pdf_bytes, max_pages=max_chunks * pages_per_chunk)
                
                if not relevant_pages:
                    logger.warning("No relevant pages identified, falling back to first pages")
                    relevant_pages = list(range(min(max_chunks * pages_per_chunk, total_pages)))
                
                # Group relevant pages into chunks
                chunks = []
                i = 0
                while i < len(relevant_pages) and len(chunks) < max_chunks:
                    # Start chunk at current relevant page
                    start_page = relevant_pages[i]
                    
                    # Include consecutive relevant pages in the same chunk
                    end_idx = min(i + pages_per_chunk, len(relevant_pages))
                    end_page = relevant_pages[end_idx - 1] + 1  # Exclusive end
                    
                    chunks.append((start_page, end_page))
                    i = end_idx
                
                logger.info(f"Smart chunking: Split PDF into {len(chunks)} prioritized chunks "
                          f"from {len(relevant_pages)} relevant pages (out of {total_pages} total pages)")
                return chunks
            else:
                # Original fixed-size chunking
                chunks = []
                for start in range(0, total_pages, pages_per_chunk):
                    end = min(start + pages_per_chunk, total_pages)
                    chunks.append((start, end))
                
                logger.info(f"Split PDF into {len(chunks)} chunks ({pages_per_chunk} pages per chunk, {total_pages} total pages)")
                return chunks
            
        except Exception as e:
            logger.error(f"Error splitting PDF into chunks: {e}")
            return []
    
    def extract_pdf_pages_as_bytes(self, pdf_bytes: bytes, page_start: int, page_end: int) -> Optional[bytes]:
        """
        Extract a range of PDF pages and return as a new PDF containing only those pages.
        
        Args:
            pdf_bytes: Original PDF file as bytes
            page_start: Starting page number (0-indexed)
            page_end: Ending page number (exclusive, 0-indexed)
            
        Returns:
            PDF bytes containing only the specified pages, or None if extraction fails
        """
        try:
            # Open original PDF
            original_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(original_doc)
            
            # Clamp page range to valid pages
            page_start = max(0, min(page_start, total_pages - 1))
            page_end = max(page_start + 1, min(page_end, total_pages))
            
            # Create new PDF with selected pages
            new_doc = fitz.open()  # Create empty PDF
            
            for page_num in range(page_start, page_end):
                new_doc.insert_pdf(original_doc, from_page=page_num, to_page=page_num)
            
            # Convert to bytes
            pdf_bytes_out = new_doc.tobytes()
            
            original_doc.close()
            new_doc.close()
            
            logger.debug(f"Extracted pages {page_start + 1}-{page_end} as PDF ({len(pdf_bytes_out)} bytes)")
            return pdf_bytes_out
            
        except Exception as e:
            logger.error(f"Error extracting PDF pages as bytes: {e}")
            return None
    
    def pdf_to_base64(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF bytes to base64-encoded string for Bedrock API.
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Base64-encoded PDF string
        """
        return base64.b64encode(pdf_bytes).decode('utf-8')
    
    def process_pdf_chunk(self, pdf_bytes: bytes, page_start: int, page_end: int) -> Optional[str]:
        """
        Process a PDF chunk: extract pages and convert to base64-encoded PDF.
        
        Args:
            pdf_bytes: PDF file as bytes
            page_start: Starting page number (0-indexed)
            page_end: Ending page number (exclusive, 0-indexed)
            
        Returns:
            Base64-encoded PDF string, or None if processing fails
        """
        # Extract pages as PDF bytes
        pdf_chunk_bytes = self.extract_pdf_pages_as_bytes(pdf_bytes, page_start, page_end)
        if not pdf_chunk_bytes:
            return None
        
        # Convert to base64
        base64_pdf = self.pdf_to_base64(pdf_chunk_bytes)
        return base64_pdf

