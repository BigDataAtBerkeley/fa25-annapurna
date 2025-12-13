"""
PDF processing utilities for extracting page chunks and converting to PDF bytes.
Uses a trained classifier to determine which pages are relevant for code generation before sending to Claude
"""

import base64
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

from page_classifier import PageRelevanceClassifier
CLASSIFIER_AVAILABLE = True

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception as e:
    PYMUPDF_AVAILABLE = False
    logger.error(f"PyMuPDF unavailable: {e}")



class PDFProcessor:
    """Process PDFs and select relevant pages for Claude for code ten"""

    def __init__(self, aws_region: str = "us-east-1", use_classifier: bool = True):
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required (pip install pymupdf)")

        self.aws_region = aws_region
        self.s3_client = boto3.client("s3", region_name=aws_region)

        self.use_classifier = bool(use_classifier and CLASSIFIER_AVAILABLE)
        self._classifier: Optional[PageRelevanceClassifier] = None
        self._classifier_enabled = self.use_classifier

        logger.info(
            "PDFProcessor initialized (classifier_enabled=%s)",
            self._classifier_enabled,
        )

    # Lazy classifier load
    @property
    def classifier(self) -> Optional[PageRelevanceClassifier]:
        if not self._classifier_enabled:
            return None

        if self._classifier is None:
            try:
                logger.info("Lazy-loading page relevance classifier...")
                self._classifier = PageRelevanceClassifier()
                if not self._classifier.is_trained:
                    logger.warning("Classifier loaded but not trained; disabling.")
                    self._classifier_enabled = False
                    self._classifier = None
            except Exception as e:
                logger.error(f"Classifier load failed: {e}")
                self._classifier_enabled = False
                self._classifier = None

        return self._classifier

    # S3 helpers
    def download_pdf_from_s3(self, bucket: str, key: str) -> Optional[bytes]:
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return None

    @staticmethod
    def is_pdf_file(key: str) -> bool:
        return key.lower().endswith(".pdf")


    def analyze_page_relevance(
        self, page: "fitz.Page", page_num: int
    ) -> Dict[str, Any]:
        text = page.get_text() or ""
        text_lower = text.lower()

        if self.classifier:
            try:
                pred = self.classifier.predict(text)
                confidence = pred["probability"]
                return {
                    "score": confidence * 100.0,
                    "is_relevant": pred["is_relevant"],
                    "confidence": confidence,
                    "method": "classifier",
                }
            except Exception as e:
                logger.warning(f"Classifier failed on page {page_num}: {e}")

        score = 0.0

        if page_num < 3 and "abstract" in text_lower:
            score += 10

        math_hits = len(re.findall(r"[=∑∫αβγθλμσπ∞]", text))
        score += min(math_hits * 2.0, 15.0)

        if page.get_images() or page.get_drawings():
            score += 8.0

        key_terms = [
            "algorithm",
            "architecture",
            "model",
            "implementation",
            "pseudocode",
            "training",
            "loss",
            "optimizer",
        ]
        score += min(sum(t in text_lower for t in key_terms) * 1.5, 10.0)

        is_relevant = score >= 10.0

        return {
            "score": score,
            "is_relevant": is_relevant,
            "method": "heuristic",
        }


    def identify_relevant_pages(
        self, pdf_bytes: bytes, max_pages: int = 20
    ) -> List[int]:

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        page_results = []
        for i in range(total_pages):
            res = self.analyze_page_relevance(doc[i], i)
            page_results.append((i, res["score"], res["is_relevant"]))

        doc.close()

        relevant = [p for p, _, r in page_results if r]

        # Fallback: always include first two pages
        if not relevant:
            relevant = list(range(min(2, total_pages)))

        # Sort by score desc, cap
        relevant = sorted(
            relevant,
            key=lambda p: page_results[p][1],
            reverse=True,
        )[:max_pages]

        # Preserve document order
        return sorted(set(relevant))


    def split_pdf_into_chunks(
        self,
        pdf_bytes: bytes,
        pages_per_chunk: int = 2,
        use_smart_chunking: bool = True,
        max_chunks: int = 15,
    ) -> List[Tuple[int, int]]:

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()

        if not use_smart_chunking:
            return [
                (i, min(i + pages_per_chunk, total_pages))
                for i in range(0, total_pages, pages_per_chunk)
            ]

        relevant_pages = self.identify_relevant_pages(
            pdf_bytes, max_pages=max_chunks * pages_per_chunk
        )

        chunks = []
        i = 0
        while i < len(relevant_pages) and len(chunks) < max_chunks:
            start = relevant_pages[i]
            end = start + 1

            while (
                i + 1 < len(relevant_pages)
                and relevant_pages[i + 1] == end
                and (end - start) < pages_per_chunk
            ):
                i += 1
                end += 1

            chunks.append((start, end))
            i += 1

        return chunks


    # PDF extraction helpers
    def extract_pdf_pages_as_bytes(
        self, pdf_bytes: bytes, start: int, end: int
    ) -> Optional[bytes]:
        try:
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
            dst = fitz.open()

            for p in range(start, min(end, len(src))):
                dst.insert_pdf(src, from_page=p, to_page=p)

            out = dst.tobytes()
            src.close()
            dst.close()
            return out
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return None

    @staticmethod
    def pdf_to_base64(pdf_bytes: bytes) -> str:
        return base64.b64encode(pdf_bytes).decode("utf-8")

    def process_pdf_chunk(
        self, pdf_bytes: bytes, start: int, end: int
    ) -> Optional[str]:
        chunk = self.extract_pdf_pages_as_bytes(pdf_bytes, start, end)
        if not chunk:
            return None
        return self.pdf_to_base64(chunk)
