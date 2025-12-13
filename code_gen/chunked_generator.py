"""
Chunked PyTorch code generator.
Splits papers into chunks, generates summaries, then combines for final code generation.
Coordinates the code gen process (uses `chunked_bedrock_client.py` to make Bedrock calls)
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from opensearch_client import OpenSearchClient
from dataset_recommender import DatasetRecommender
from pdf_processor import PDFProcessor
from chunked_bedrock_client import ChunkedBedrockClient

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)



class ChunkedPyTorchGenerator:
    """Generator that processes papers in chunks to handle long papers."""

    def __init__(
        self,
        batch_size: int = 8,  # Number of chunk summaries to combine in each batch when recursively merging
        pages_per_pdf_chunk: int = 4,  # Number of PDF pages per chunk when splitting the PDF
        max_pdf_chunks: int = 15,  # Max number of PDF chunks to process
    ):

        self.opensearch_client = OpenSearchClient() 
        self.chunked_bedrock_client = ChunkedBedrockClient()
        self.dataset_recommender = DatasetRecommender(bedrock_client=self.chunked_bedrock_client)

        self.batch_size = batch_size
        self.pages_per_pdf_chunk = pages_per_pdf_chunk
        self.max_pdf_chunks = max_pdf_chunks

        self.pdf_processor = PDFProcessor(
            aws_region=self.opensearch_client.aws_region,
            use_classifier=True # classifier for page relevance
        )
        logger.info("PDF processor initialized - PDF document processing with classifier enabled")

        # Results directory for saving chunk results locally (gitignored)
        self.results_base_dir = Path(__file__).parent.parent / "results"

        logger.info("Chunked PyTorch Code Generator initialized: ")
        logger.info(f"  - Batch size for recursive merging: {batch_size}")
        logger.info(f"  - Pages per PDF chunk: {pages_per_pdf_chunk}")

    def _process_pdf_chunk(
        self,
        pdf_bytes: bytes,
        page_start: int,
        page_end: int,
        chunk_number: int,
        total_chunks: int,
        paper_summary: str,
    ) -> Dict[str, Any]:
        """
        Process a single PDF chunk using PDF document input.
        """
        chunk_start = time.time()
        try:
            base64_pdf = self.pdf_processor.process_pdf_chunk(pdf_bytes, page_start, page_end)

            if not base64_pdf:
                return {
                    "chunk_number": chunk_number,
                    "success": False,
                    "error": "Failed to extract PDF pages",
                    "processing_time": time.time() - chunk_start,
                }

            result = self.chunked_bedrock_client.summarize_pdf_chunk(
                base64_pdf=base64_pdf,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                paper_summary=paper_summary,
                page_start=page_start,
                page_end=page_end,
            )

            if result.get("success"):
                return {
                    "chunk_number": chunk_number,
                    "success": True,
                    "summary": result["summary"],
                    "summary_length": len(result["summary"]),
                    "processing_time": time.time() - chunk_start,
                    "pages": result.get("pages", f"{page_start + 1}-{page_end}"),
                    "num_pages": result.get("num_pages", page_end - page_start),
                    "chunk_type": "pdf_document",
                }
            else:
                return {
                    "chunk_number": chunk_number,
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "processing_time": time.time() - chunk_start,
                }
        except Exception as e:
            logger.error(f"Error processing PDF chunk {chunk_number}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return {
                "chunk_number": chunk_number,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - chunk_start,
            }

    def generate_code_for_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Generate PyTorch code for a paper using chunked approach
        """
        start_time = time.time()

        try:
            logger.info(f"Generating code for paper ID: {paper_id}")

            # Step 1: Retrieve paper from OpenSearch
            paper = self.opensearch_client.get_paper_by_id(paper_id)
            if not paper:
                return {
                    "success": False,
                    "error": f"Paper with ID {paper_id} not found",
                    "paper_id": paper_id,
                }

            paper_summary = self.opensearch_client.get_paper_summary(paper) # summary of paper alrdy indexed in OpenSearch

            # Get PDF bytes
            pdf_bytes = self.opensearch_client.get_paper_pdf_bytes(paper)
            if not pdf_bytes:
                return {
                    "success": False,
                    "error": f"Failed to retrieve PDF for paper {paper_id}",
                    "paper_id": paper_id,
                }

            # Step 2: Split PDF into page chunks
            # 1. Analyzing each page for relevance (formulas, diagrams, key terms, section headers)
            # 2. Scoring pages (methodology=8pts, architecture=10pts, implementation=9pts, references=0pts)
            # 3. Selecting top N most relevant pages
            # 4. Grouping them into chunks of pages_per_pdf_chunk size
            pdf_chunks = self.pdf_processor.split_pdf_into_chunks(
                pdf_bytes,
                pages_per_chunk=self.pages_per_pdf_chunk,
                use_smart_chunking=True,
                max_chunks=self.max_pdf_chunks,
            )

            if not pdf_chunks:
                return {
                    "success": False,
                    "error": "Failed to split PDF into chunks",
                    "paper_id": paper_id,
                }

            logger.info(
                f"Split PDF into {len(pdf_chunks)} chunks "
                f"({self.pages_per_pdf_chunk} pages per chunk)"
            )

            # Step 3: Generate summaries for each PDF chunk
            chunk_summaries: List[str] = []
            chunk_results: List[Dict[str, Any]] = []
            num_chunks = len(pdf_chunks)

            if num_chunks > 0:
                initial_delay = self.chunked_bedrock_client.chunk_delay
                logger.info(f"Waiting {initial_delay}s before starting PDF chunk processing")
                time.sleep(initial_delay)

            for i, (page_start, page_end) in enumerate(pdf_chunks, 1):
                logger.info(
                    f"Processing PDF chunk {i}/{num_chunks} "
                    f"(pages {page_start + 1}-{page_end})..."
                )

                result = self._process_pdf_chunk(
                    pdf_bytes, page_start, page_end, i, num_chunks, paper_summary
                )

                if result.get("success"):
                    chunk_summaries.append(result["summary"])
                    logger.info(
                        f"PDF Chunk {i} summarized ({result['summary_length']:,} chars "
                    )
                else:
                    logger.error(
                        f"PDF Chunk {i} summarization failed: {result.get('error')}"
                    )

                chunk_results.append(result)

                # Add delay between chunks to avoid throttling
                if i < num_chunks:
                    delay = self.chunked_bedrock_client.chunk_delay
                    logger.info(
                        f"Waiting {delay}s before processing next PDF chunk "
                        "(throttling mitigation)..."
                    )
                    time.sleep(delay)

            successful_chunks = sum(1 for r in chunk_results if r.get("success"))
            if successful_chunks < num_chunks // 2:
                return {
                    "success": False,
                    "error": (
                        "Too many chunk summarization failures "
                        f"({successful_chunks}/{num_chunks} succeeded)"
                    ),
                    "paper_id": paper_id,
                    "chunk_results": chunk_results,
                }

            logger.info(
                f"Generated {successful_chunks}/{num_chunks} chunk summaries"
            )

            # Save chunk results locally to results folder (gitignored)
            self._save_chunk_results(paper_id, chunk_results, chunk_summaries)

            context_for_dataset = paper_summary
            if chunk_summaries:
                early_chunks = "\n\n".join(
                    chunk_summaries[: min(5, len(chunk_summaries))]
                )
                context_for_dataset = (f"{paper_summary}\n\nAdditional Context from Paper:\n{early_chunks}")

            dataset_recommendations = self.dataset_recommender.recommend_datasets(
                paper, context_for_dataset, use_llm=True
            )
            logger.info(
                "Recommended datasets: "
                f"{dataset_recommendations.get('recommended_datasets', [])}"
            )

            # Step 5: Two-stage hierarchical summarization
            # Stage 5a: Group chunk summaries into batches and summarize each batch
            logger.info(
                f"Summarizing {len(chunk_summaries)} chunk summaries into batches of "
                f"{self.batch_size}..."
            )
            batch_summaries = self._summarize_batches(chunk_summaries, paper_summary)

            if not batch_summaries:
                return {
                    "success": False,
                    "error": "Failed to generate batch summaries",
                    "paper_id": paper_id,
                    "chunk_results": chunk_results,
                }

            logger.info(f"Generated {len(batch_summaries)} batch summaries")

            # Step 5b: Generate final code from batch summaries
            logger.info("Stage 5b: Generating final PyTorch code")
            final_start = time.time()

            try:
                final_result = self.chunked_bedrock_client.generate_final_code(
                    paper_summary=paper_summary,
                    chunk_summaries=batch_summaries,
                    dataset_recommendations=dataset_recommendations,
                )

                if not final_result.get("success") or not final_result.get("code"):
                    return {
                        "success": False,
                        "error": final_result.get("error", "Final code generation failed"),
                        "paper_id": paper_id,
                        "chunk_results": chunk_results,
                    }

                logger.info(f"✅ Final code generated ({len(final_result['code']):,} chars)")

                # Combine results - code review happens separately in pipeline/lambda handler
                result: Dict[str, Any] = {
                    "success": True,
                    "paper_id": paper_id,
                    "paper_title": paper.get("title", "Unknown"),
                    "paper_authors": paper.get("authors", []),
                    "code": final_result["code"],
                    "model_used": final_result["model_used"],
                    "generated_at": datetime.now().isoformat(),
                    "dataset_recommendations": dataset_recommendations,
                    "recommended_dataset": dataset_recommendations.get(
                        "primary_dataset", "synthetic"
                    ),
                    "code_review": {
                        "fixes_applied": [],
                        "iterations": 0,
                        "review_time": 0.0,
                    },
                    "chunk_results": chunk_results,
                    "num_chunks": num_chunks,
                    "successful_chunks": successful_chunks,
                    "num_batch_summaries": len(batch_summaries),
                    "batch_size": self.batch_size,
                    "pages_per_pdf_chunk": self.pages_per_pdf_chunk,
                    "total_generation_time": time.time() - start_time,
                    "final_generation_time": time.time() - final_start,
                }

                logger.info(f"Successfully generated code in {result['total_generation_time']:.1f}s")
                return result

            except Exception as e:
                logger.error(f"Error in final code generation: {e}")
                return {
                    "success": False,
                    "error": f"Final code generation error: {str(e)}",
                    "paper_id": paper_id,
                    "chunk_results": chunk_results,
                }

        except Exception as e:
            logger.error(f"Error generating code for paper {paper_id}: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "paper_id": paper_id,
            }

    def _summarize_batches(
        self, chunk_summaries: List[str], paper_summary: str
    ) -> List[str]:
        """
        Summarize chunk summaries into batches. Returns: List of batch summaries
        """
        if not chunk_summaries:
            return []

        # Group chunk summaries into batches
        batches: List[List[str]] = []
        for i in range(0, len(chunk_summaries), self.batch_size):
            batch = chunk_summaries[i : i + self.batch_size]
            batches.append(batch)

        total_batches = len(batches)

        batch_summaries: List[str] = []
        for i, batch in enumerate(batches, 1):
            logger.info(
                f"Summarizing batch {i}/{total_batches} ({len(batch)} chunk summaries)"
            )
            batch_start = time.time()

            try:
                result = self.chunked_bedrock_client.summarize_batch(
                    batch_summaries=batch,
                    batch_number=i,
                    total_batches=total_batches,
                    paper_summary=paper_summary,
                )

                if result.get("success"):
                    batch_summaries.append(result["summary"])
                    logger.info(
                        f"Batch {i} summarized "
                        f"({len(result['summary']):,} chars, {time.time() - batch_start:.1f}s)"
                    )
                else:
                    logger.error(
                        f"Batch {i} summarization failed: {result.get('error')}"
                    )

            except Exception as e:
                logger.error(f"Error summarizing batch {i}: {e}")

            # delay between batches to avoid throttling
            if i < total_batches:
                delay = self.chunked_bedrock_client.chunk_delay
                logger.info(
                    f"Waiting {delay}s before processing next batch "
                    "(throttling mitigation)..."
                )
                time.sleep(delay)

        return batch_summaries

    # IGNORE (Only used locally and for testing)
    def _save_chunk_results(
        self, paper_id: str, chunk_results: List[Dict[str, Any]], chunk_summaries: List[str]
    ) -> None:
        """
        Save chunk results to results directory locally (gitignored).
        """
        try:
            chunk_results_dir = self.results_base_dir / paper_id / "chunk-results"
            chunk_results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            all_chunks_file = chunk_results_dir / f"{paper_id}_all_chunks_{timestamp}.json"
            chunk_data: Dict[str, Any] = {
                "paper_id": paper_id,
                "timestamp": timestamp,
                "total_chunks": len(chunk_results),
                "successful_chunks": sum(
                    1 for r in chunk_results if r.get("success")
                ),
                "chunks": [],
            }

            summary_idx = 0
            for chunk_result in chunk_results:
                chunk_data_entry = chunk_result.copy()
                if chunk_result.get("success") and summary_idx < len(chunk_summaries):
                    chunk_data_entry["summary"] = chunk_summaries[summary_idx]
                    summary_idx += 1
                chunk_data["chunks"].append(chunk_data_entry)

            with open(all_chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved chunk results to {all_chunks_file}")

            for i, chunk_result in enumerate(chunk_results, 1):
                if chunk_result.get("success") and i <= len(chunk_summaries):
                    chunk_file = (
                        chunk_results_dir / f"{paper_id}_chunk_{i:03d}_{timestamp}.json"
                    )
                    chunk_entry = {
                        "paper_id": paper_id,
                        "chunk_number": i,
                        "timestamp": timestamp,
                        **chunk_result,
                        "summary": chunk_summaries[i - 1],
                    }
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        json.dump(chunk_entry, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to save chunk results: {e}")

