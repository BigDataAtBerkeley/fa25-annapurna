"""
Chunked PyTorch code generator.
Splits papers into chunks, generates summaries, then combines for final code generation.
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from parent directory (code-gen-for-deliv)
try:
    from opensearch_client import OpenSearchClient
    from dataset_recommender import DatasetRecommender
    from code_review_agent import CodeReviewAgent
except ImportError:
    # Try relative import if in package
    try:
        from ..opensearch_client import OpenSearchClient
        from ..dataset_recommender import DatasetRecommender
        from ..code_review_agent import CodeReviewAgent
    except ImportError:
        # Fallback: try direct import from parent
        import importlib.util
        opensearch_path = os.path.join(parent_dir, 'opensearch_client.py')
        dataset_path = os.path.join(parent_dir, 'dataset_recommender.py')
        code_review_path = os.path.join(parent_dir, 'code_review_agent.py')
        if os.path.exists(opensearch_path):
            spec = importlib.util.spec_from_file_location("opensearch_client", opensearch_path)
            opensearch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(opensearch_module)
            OpenSearchClient = opensearch_module.OpenSearchClient
        if os.path.exists(dataset_path):
            spec = importlib.util.spec_from_file_location("dataset_recommender", dataset_path)
            dataset_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset_module)
            DatasetRecommender = dataset_module.DatasetRecommender
        if os.path.exists(code_review_path):
            spec = importlib.util.spec_from_file_location("code_review_agent", code_review_path)
            code_review_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(code_review_module)
            CodeReviewAgent = code_review_module.CodeReviewAgent

# Import from same directory
try:
    from chunked_bedrock_client import ChunkedBedrockClient
except ImportError:
    from .chunked_bedrock_client import ChunkedBedrockClient

# Import PDF processor from parent directory
try:
    from pdf_processor import PDFProcessor
except ImportError:
    # Try relative import
    try:
        from ..pdf_processor import PDFProcessor
    except ImportError:
        # Fallback: try direct import from parent
        import importlib.util
        pdf_processor_path = os.path.join(parent_dir, 'pdf_processor.py')
        if os.path.exists(pdf_processor_path):
            spec = importlib.util.spec_from_file_location("pdf_processor", pdf_processor_path)
            pdf_processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pdf_processor_module)
            PDFProcessor = pdf_processor_module.PDFProcessor
        else:
            PDFProcessor = None
            logger.warning("PDF processor not available - PDF processing will be disabled")

logger = logging.getLogger(__name__)

class ChunkedPyTorchGenerator:
    """Generator that processes papers in chunks to handle long papers."""
    
    def __init__(self, max_chunk_size: int = 150000, use_haiku_for_chunks: bool = True, 
                 parallel_chunks: bool = False, max_parallel: int = 2, batch_size: int = 8,
                 pages_per_pdf_chunk: int = 2, use_smart_pdf_chunking: bool = True, 
                 max_pdf_chunks: int = 15):
        """
        Initialize chunked code generator.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters (default: 150k)
                           Chunks will be created to fit within this size
                           Note: Reduced from 200k to account for prompt + paper summary overhead
            use_haiku_for_chunks: Use Claude 3 Haiku for chunk summaries (faster, better rate limits)
            parallel_chunks: If True, process chunks in parallel (default: False, sequential)
            max_parallel: Maximum number of chunks to process in parallel (default: 2)
            batch_size: Number of chunk summaries to combine in each batch for hierarchical summarization (default: 8)
            pages_per_pdf_chunk: Number of pages per PDF chunk when processing PDFs (default: 2)
            use_smart_pdf_chunking: If True, only process relevant PDF pages (abstract, formulas, diagrams, etc.) (default: True)
            max_pdf_chunks: Maximum number of PDF chunks to process when using smart chunking (default: 15)
        """
        self.opensearch_client = OpenSearchClient()
        self.chunked_bedrock_client = ChunkedBedrockClient(use_haiku=use_haiku_for_chunks)
        self.dataset_recommender = DatasetRecommender(bedrock_client=None)  # Don't need bedrock for dataset rec
        # Initialize code review agent (use the bedrock client from chunked_bedrock_client if needed)
        # CodeReviewAgent will create its own BedrockClient if not provided
        # Enable execution testing if requested
        enable_execution_testing = os.getenv('ENABLE_EXECUTION_TESTING', 'false').lower() == 'true'
        self.code_review_agent = CodeReviewAgent(
            bedrock_client=None,
            enable_execution_testing=enable_execution_testing
        )
        self.max_chunk_size = max_chunk_size
        self.parallel_chunks = parallel_chunks
        self.max_parallel = max_parallel
        self.batch_size = batch_size
        self.pages_per_pdf_chunk = pages_per_pdf_chunk
        self.use_smart_pdf_chunking = use_smart_pdf_chunking
        self.max_pdf_chunks = max_pdf_chunks
        
        # Initialize PDF processor if available
        try:
            if PDFProcessor:
                self.pdf_processor = PDFProcessor(aws_region=self.opensearch_client.aws_region)
                logger.info("PDF processor initialized - PDF vision processing enabled")
            else:
                self.pdf_processor = None
                logger.warning("PDF processor not available - PDF processing disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize PDF processor: {e} - PDF processing disabled")
            self.pdf_processor = None
        
        # Results directory for saving chunk results
        # Path: midpoint-deliverable/results/{paper_id}/chunk-results/
        # Go up 3 levels: chunked-code-gen -> code-gen-for-deliv -> midpoint-deliverable
        self.results_base_dir = Path(__file__).parent.parent.parent / 'results'
        
        logger.info(f"Chunked PyTorch Code Generator initialized:")
        logger.info(f"  - Max chunk size: {max_chunk_size:,} characters")
        logger.info(f"  - Use Haiku for chunks: {use_haiku_for_chunks}")
        logger.info(f"  - Parallel processing: {parallel_chunks} (max {max_parallel} concurrent)")
        logger.info(f"  - Batch size for hierarchical summarization: {batch_size}")
        logger.info(f"  - Pages per PDF chunk: {pages_per_pdf_chunk}")
        logger.info(f"  - Smart PDF chunking: {use_smart_pdf_chunking} (max {max_pdf_chunks} chunks)")
        logger.info(f"  - PDF vision processing: {'enabled' if self.pdf_processor else 'disabled'}")
    
    def _process_pdf_chunk(self, pdf_bytes: bytes, page_start: int, page_end: int, 
                           chunk_number: int, total_chunks: int, paper_summary: str) -> Dict[str, Any]:
        """
        Process a single PDF chunk using vision capabilities.
        
        Args:
            pdf_bytes: PDF file as bytes
            page_start: Starting page number (0-indexed)
            page_end: Ending page number (exclusive, 0-indexed)
            chunk_number: Chunk number (1-indexed)
            total_chunks: Total number of chunks
            paper_summary: Paper summary
            
        Returns:
            Dictionary with chunk result
        """
        chunk_start = time.time()
        try:
            # Extract PDF pages as images
            base64_images = self.pdf_processor.process_pdf_chunk(
                pdf_bytes, page_start, page_end, dpi=150
            )
            
            if not base64_images:
                return {
                    "chunk_number": chunk_number,
                    "success": False,
                    "error": "Failed to extract images from PDF pages",
                    "processing_time": time.time() - chunk_start
                }
            
            # Summarize using vision
            result = self.chunked_bedrock_client.summarize_pdf_chunk_with_vision(
                base64_images=base64_images,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                paper_summary=paper_summary,
                page_start=page_start,
                page_end=page_end
            )
            
            if result.get("success"):
                return {
                    "chunk_number": chunk_number,
                    "success": True,
                    "summary": result["summary"],
                    "summary_length": len(result["summary"]),
                    "processing_time": time.time() - chunk_start,
                    "pages": result.get("pages", f"{page_start + 1}-{page_end}"),
                    "num_images": result.get("num_images", len(base64_images)),
                    "chunk_type": "pdf_vision"
                }
            else:
                return {
                    "chunk_number": chunk_number,
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "processing_time": time.time() - chunk_start
                }
        except Exception as e:
            logger.error(f"Error processing PDF chunk {chunk_number}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                "chunk_number": chunk_number,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - chunk_start
            }
    
    def _process_chunk(self, chunk: str, chunk_number: int, total_chunks: int, 
                      paper_summary: str) -> Dict[str, Any]:
        """
        Process a single text chunk (helper method for parallel processing).
        
        Args:
            chunk: Chunk text
            chunk_number: Chunk number (1-indexed)
            total_chunks: Total number of chunks
            paper_summary: Paper summary
            
        Returns:
            Dictionary with chunk result
        """
        chunk_start = time.time()
        try:
            result = self.chunked_bedrock_client.summarize_chunk(
                chunk_text=chunk,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                paper_summary=paper_summary
            )
            
            if result.get("success"):
                return {
                    "chunk_number": chunk_number,
                    "success": True,
                    "summary": result["summary"],
                    "summary_length": len(result["summary"]),
                    "processing_time": time.time() - chunk_start,
                    "chunk_type": "text"
                }
            else:
                return {
                    "chunk_number": chunk_number,
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "processing_time": time.time() - chunk_start
                }
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_number}: {e}")
            return {
                "chunk_number": chunk_number,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - chunk_start
            }
    
    def _process_chunks_sequential(self, chunks: List[str], paper_summary: str) -> tuple:
        """
        Process chunks sequentially with delays.
        
        Returns:
            Tuple of (chunk_summaries list, chunk_results list)
        """
        chunk_summaries = []
        chunk_results = []
        num_chunks = len(chunks)
        
        # Add initial delay before first chunk to avoid immediate throttling
        if num_chunks > 0:
            initial_delay = self.chunked_bedrock_client.chunk_delay
            logger.info(f"Waiting {initial_delay}s before starting chunk processing (throttling mitigation)...")
            time.sleep(initial_delay)
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{num_chunks}...")
            
            result = self._process_chunk(chunk, i, num_chunks, paper_summary)
            
            if result.get("success"):
                chunk_summaries.append(result["summary"])
                logger.info(f"✅ Chunk {i} summarized ({result['summary_length']:,} chars)")
            else:
                logger.error(f"❌ Chunk {i} summarization failed: {result.get('error')}")
            
            chunk_results.append(result)
            
            # Add delay between chunks to avoid throttling
            if i < num_chunks:
                delay = self.chunked_bedrock_client.chunk_delay
                logger.info(f"Waiting {delay}s before processing next chunk (throttling mitigation)...")
                time.sleep(delay)
        
        return chunk_summaries, chunk_results
    
    def _process_chunks_parallel(self, chunks: List[str], paper_summary: str) -> tuple:
        """
        Process chunks in parallel with rate limiting.
        
        Returns:
            Tuple of (chunk_summaries list, chunk_results list)
        """
        chunk_summaries = []
        chunk_results = [None] * len(chunks)  # Pre-allocate to maintain order
        
        # Process chunks in parallel with max_parallel limit
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(
                    self._process_chunk, 
                    chunk, 
                    i+1, 
                    len(chunks), 
                    paper_summary
                ): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results[chunk_index] = result
                    
                    if result.get("success"):
                        chunk_summaries.append(result["summary"])
                        logger.info(f"✅ Chunk {result['chunk_number']} completed ({result.get('summary_length', 0):,} chars)")
                    else:
                        logger.error(f"❌ Chunk {result['chunk_number']} failed: {result.get('error')}")
                except Exception as e:
                    logger.error(f"Chunk {chunk_index + 1} raised exception: {e}")
                    chunk_results[chunk_index] = {
                        "chunk_number": chunk_index + 1,
                        "success": False,
                        "error": str(e),
                        "processing_time": 0
                    }
        
        # Sort chunk_results by chunk_number to maintain order
        chunk_results = sorted([r for r in chunk_results if r is not None], 
                             key=lambda x: x.get("chunk_number", 0))
        
        return chunk_summaries, chunk_results
    
    def split_into_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        """
        Split text into chunks of maximum size.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum size of each chunk in characters
            
        Returns:
            List of text chunks (each <= max_chunk_size)
        """
        if not text:
            return []
        
        total_length = len(text)
        chunks = []
        
        # Split into chunks of max_chunk_size
        for i in range(0, total_length, max_chunk_size):
            chunk = text[i:i + max_chunk_size]
            chunks.append(chunk)
        
        num_chunks = len(chunks)
        logger.info(f"Split paper into {num_chunks} chunks (max {max_chunk_size:,} chars each): {[len(c) for c in chunks]} characters each")
        
        return chunks
    
    def generate_code_for_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Generate PyTorch code for a paper using chunked approach.
        
        Args:
            paper_id: OpenSearch document ID
            
        Returns:
            Dictionary containing generated code and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Generating code for paper ID: {paper_id} (chunked approach)")
            
            # Step 1: Retrieve paper
            paper = self.opensearch_client.get_paper_by_id(paper_id)
            if not paper:
                return {
                    "success": False,
                    "error": f"Paper with ID {paper_id} not found",
                    "paper_id": paper_id
                }
            
            paper_summary = self.opensearch_client.get_paper_summary(paper)
            
            # Check if paper is a PDF and PDF processing is available
            is_pdf = self.opensearch_client.is_pdf_paper(paper)
            use_pdf_vision = is_pdf and self.pdf_processor is not None
            
            if use_pdf_vision:
                logger.info(f"📄 Paper is a PDF - using vision-based processing to extract formulas and diagrams")
                # Get PDF bytes
                pdf_bytes = self.opensearch_client.get_paper_pdf_bytes(paper)
                if not pdf_bytes:
                    return {
                        "success": False,
                        "error": f"Failed to retrieve PDF for paper {paper_id}",
                        "paper_id": paper_id
                    }
                
                # Step 3: Split PDF into page chunks (using smart chunking to prioritize relevant sections)
                pdf_chunks = self.pdf_processor.split_pdf_into_chunks(
                    pdf_bytes, 
                    pages_per_chunk=self.pages_per_pdf_chunk,
                    use_smart_chunking=self.use_smart_pdf_chunking,
                    max_chunks=self.max_pdf_chunks
                )
                
                if not pdf_chunks:
                    return {
                        "success": False,
                        "error": "Failed to split PDF into chunks",
                        "paper_id": paper_id
                    }
                
                logger.info(f"Split PDF into {len(pdf_chunks)} chunks ({self.pages_per_pdf_chunk} pages per chunk)")
                
                # Step 4: Generate summaries for each PDF chunk using vision
                chunk_summaries = []
                chunk_results = []
                num_chunks = len(pdf_chunks)
                
                # Add initial delay before first chunk
                if num_chunks > 0:
                    initial_delay = self.chunked_bedrock_client.chunk_delay
                    logger.info(f"Waiting {initial_delay}s before starting PDF chunk processing (throttling mitigation)...")
                    time.sleep(initial_delay)
                
                for i, (page_start, page_end) in enumerate(pdf_chunks, 1):
                    logger.info(f"Processing PDF chunk {i}/{num_chunks} (pages {page_start + 1}-{page_end})...")
                    
                    result = self._process_pdf_chunk(
                        pdf_bytes, page_start, page_end, i, num_chunks, paper_summary
                    )
                    
                    if result.get("success"):
                        chunk_summaries.append(result["summary"])
                        logger.info(f"✅ PDF Chunk {i} summarized ({result['summary_length']:,} chars, pages {result.get('pages', 'N/A')})")
                    else:
                        logger.error(f"❌ PDF Chunk {i} summarization failed: {result.get('error')}")
                    
                    chunk_results.append(result)
                    
                    # Add delay between chunks to avoid throttling
                    if i < num_chunks:
                        delay = self.chunked_bedrock_client.chunk_delay
                        logger.info(f"Waiting {delay}s before processing next PDF chunk (throttling mitigation)...")
                        time.sleep(delay)
            else:
                # Fall back to text-based processing
                if is_pdf:
                    logger.warning("Paper is a PDF but PDF processor not available - falling back to text extraction")
                
                paper_content = self.opensearch_client.get_paper_content(paper)
                
                if not paper_content:
                    return {
                        "success": False,
                        "error": f"Paper {paper_id} has no content",
                        "paper_id": paper_id
                    }
                
                logger.info(f"Paper content length: {len(paper_content):,} characters")
                
                # Step 2: Get dataset recommendations
                dataset_recommendations = self.dataset_recommender.recommend_datasets(
                    paper, paper_content, use_llm=False  # Use rule-based for speed
                )
                logger.info(f"Recommended datasets: {dataset_recommendations.get('recommended_datasets', [])}")
                
                # Step 3: Split paper into chunks (based on max_chunk_size)
                chunks = self.split_into_chunks(paper_content, self.max_chunk_size)
                
                if not chunks:
                    return {
                        "success": False,
                        "error": "Failed to split paper into chunks",
                        "paper_id": paper_id
                    }
                
                # Step 4: Generate summaries for each chunk
                chunk_summaries = []
                chunk_results = []
                
                num_chunks = len(chunks)
                if self.parallel_chunks:
                    # Process chunks in parallel (with rate limiting)
                    logger.info(f"Processing {num_chunks} chunks in parallel (max {self.max_parallel} concurrent)...")
                    chunk_summaries, chunk_results = self._process_chunks_parallel(
                        chunks, paper_summary
                    )
                else:
                    # Process chunks sequentially (with delays to avoid throttling)
                    logger.info(f"Processing {num_chunks} chunks sequentially...")
                    chunk_summaries, chunk_results = self._process_chunks_sequential(
                        chunks, paper_summary
                    )
            
            # Check if we got enough summaries
            successful_chunks = sum(1 for r in chunk_results if r.get("success"))
            if successful_chunks < num_chunks // 2:  # Need at least half
                return {
                    "success": False,
                    "error": f"Too many chunk summarization failures ({successful_chunks}/{num_chunks} succeeded)",
                    "paper_id": paper_id,
                    "chunk_results": chunk_results
                }
            
            logger.info(f"✅ Generated {successful_chunks}/{num_chunks} chunk summaries")
            
            # Save chunk results
            self._save_chunk_results(paper_id, chunk_results, chunk_summaries)
            
            # Get dataset recommendations using chunk summaries for better context
            # Combine first few chunk summaries (usually contain abstract, intro, methods) for dataset detection
            context_for_dataset = paper_summary
            if chunk_summaries:
                # Use first 3-5 chunk summaries (typically contain methods, datasets, domain info)
                early_chunks = "\n\n".join(chunk_summaries[:min(5, len(chunk_summaries))])
                context_for_dataset = f"{paper_summary}\n\nAdditional Context from Paper:\n{early_chunks[:50000]}"  # Limit to 50k chars
            
            dataset_recommendations = self.dataset_recommender.recommend_datasets(
                paper, context_for_dataset, use_llm=False  # Use rule-based for speed
            )
            logger.info(f"Recommended datasets: {dataset_recommendations.get('recommended_datasets', [])}")
            
            # Step 5: Two-stage hierarchical summarization
            # Stage 5a: Group chunk summaries into batches and summarize each batch
            logger.info(f"Stage 5a: Summarizing {len(chunk_summaries)} chunk summaries into batches of {self.batch_size}...")
            batch_summaries = self._summarize_batches(chunk_summaries, paper_summary)
            
            if not batch_summaries:
                return {
                    "success": False,
                    "error": "Failed to generate batch summaries",
                    "paper_id": paper_id,
                    "chunk_results": chunk_results
                }
            
            logger.info(f"✅ Generated {len(batch_summaries)} batch summaries from {len(chunk_summaries)} chunk summaries")
            
            # Step 5b: Generate final code from batch summaries
            logger.info("Stage 5b: Generating final PyTorch code from batch summaries...")
            final_start = time.time()
            
            try:
                final_result = self.chunked_bedrock_client.generate_final_code(
                    paper_summary=paper_summary,
                    chunk_summaries=batch_summaries,  # Use batch summaries instead of individual chunk summaries
                    dataset_recommendations=dataset_recommendations
                )
                
                if not final_result.get("success") or not final_result.get("code"):
                    return {
                        "success": False,
                        "error": final_result.get("error", "Final code generation failed"),
                        "paper_id": paper_id,
                        "chunk_results": chunk_results
                    }
                
                logger.info(f"✅ Final code generated ({len(final_result['code']):,} chars)")
                
                # Review and fix code - REQUIRED before sending to test queue
                logger.info("Reviewing and fixing generated code...")
                primary_dataset = dataset_recommendations.get("primary_dataset", "synthetic")
                review_result = self.code_review_agent.review_and_fix_code(
                    final_result["code"],
                    dataset_name=primary_dataset,
                    paper_id=paper_id,
                    paper_title=paper.get('title', 'Unknown')
                )
                
                # Use reviewed code (even if review didn't find issues, it may have applied quick fixes)
                reviewed_code = final_result["code"]
                code_review_data = {
                    "fixes_applied": [],
                    "iterations": 0,
                    "review_time": 0.0
                }
                
                if review_result.get("code"):
                    reviewed_code = review_result["code"]
                    code_review_data = {
                        "fixes_applied": review_result.get("fixes_applied", []),
                        "iterations": review_result.get("iterations", 0),
                        "review_time": review_result.get("review_time", 0.0)
                    }
                    logger.info(f"Code review complete: {code_review_data['iterations']} iterations, "
                              f"{len(code_review_data['fixes_applied'])} fixes applied in {code_review_data['review_time']:.2f}s")
                else:
                    logger.warning("Code review returned no code, using original code")
                
                # Combine results
                result = {
                    "success": True,
                    "paper_id": paper_id,
                    "paper_title": paper.get('title', 'Unknown'),
                    "paper_authors": paper.get('authors', []),
                    "code": reviewed_code,
                    "model_used": final_result["model_used"],
                    "generated_at": datetime.now().isoformat(),
                    "dataset_recommendations": dataset_recommendations,
                    "recommended_dataset": dataset_recommendations.get("primary_dataset", "synthetic"),
                    "code_review": code_review_data,
                    "chunk_results": chunk_results,
                    "num_chunks": num_chunks,
                    "successful_chunks": successful_chunks,
                    "num_batch_summaries": len(batch_summaries),
                    "batch_size": self.batch_size,
                    "max_chunk_size": self.max_chunk_size,
                    "used_pdf_vision": use_pdf_vision,
                    "pages_per_pdf_chunk": self.pages_per_pdf_chunk if use_pdf_vision else None,
                    "total_generation_time": time.time() - start_time,
                    "final_generation_time": time.time() - final_start
                }
                
                logger.info(f"✅ Successfully generated code using chunked approach (total time: {result['total_generation_time']:.1f}s)")
                return result
                
            except Exception as e:
                logger.error(f"Error in final code generation: {e}")
                return {
                    "success": False,
                    "error": f"Final code generation error: {str(e)}",
                    "paper_id": paper_id,
                    "chunk_results": chunk_results
                }
            
        except Exception as e:
            logger.error(f"Error generating code for paper {paper_id}: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "paper_id": paper_id
            }
    
    def _summarize_batches(self, chunk_summaries: List[str], paper_summary: str) -> List[str]:
        """
        Summarize chunk summaries into batches (hierarchical summarization stage 2).
        
        Args:
            chunk_summaries: List of individual chunk summaries
            paper_summary: Overall paper summary
            
        Returns:
            List of batch summaries
        """
        if not chunk_summaries:
            return []
        
        # Group chunk summaries into batches
        batches = []
        for i in range(0, len(chunk_summaries), self.batch_size):
            batch = chunk_summaries[i:i + self.batch_size]
            batches.append(batch)
        
        total_batches = len(batches)
        logger.info(f"Grouped {len(chunk_summaries)} chunk summaries into {total_batches} batches")
        
        batch_summaries = []
        for i, batch in enumerate(batches, 1):
            logger.info(f"Summarizing batch {i}/{total_batches} ({len(batch)} chunk summaries)...")
            batch_start = time.time()
            
            try:
                result = self.chunked_bedrock_client.summarize_batch(
                    batch_summaries=batch,
                    batch_number=i,
                    total_batches=total_batches,
                    paper_summary=paper_summary
                )
                
                if result.get("success"):
                    batch_summaries.append(result["summary"])
                    logger.info(f"✅ Batch {i} summarized ({len(result['summary']):,} chars, {time.time() - batch_start:.1f}s)")
                else:
                    logger.error(f"❌ Batch {i} summarization failed: {result.get('error')}")
                    # Continue with other batches even if one fails
                    
            except Exception as e:
                logger.error(f"Error summarizing batch {i}: {e}")
            
            # Add delay between batches to avoid throttling
            if i < total_batches:
                delay = self.chunked_bedrock_client.chunk_delay
                logger.info(f"Waiting {delay}s before processing next batch (throttling mitigation)...")
                time.sleep(delay)
        
        return batch_summaries
    
    def _save_chunk_results(self, paper_id: str, chunk_results: List[Dict[str, Any]], 
                           chunk_summaries: List[str]) -> None:
        """
        Save chunk results to results directory.
        
        Args:
            paper_id: Paper ID
            chunk_results: List of chunk result dictionaries
            chunk_summaries: List of chunk summary texts
        """
        try:
            # Create chunk-results directory
            chunk_results_dir = self.results_base_dir / paper_id / 'chunk-results'
            chunk_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual chunk results with summaries
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save all chunk results in one file
            all_chunks_file = chunk_results_dir / f"{paper_id}_all_chunks_{timestamp}.json"
            chunk_data = {
                "paper_id": paper_id,
                "timestamp": timestamp,
                "total_chunks": len(chunk_results),
                "successful_chunks": sum(1 for r in chunk_results if r.get("success")),
                "chunks": []
            }
            
            # Combine chunk results with their summaries
            summary_idx = 0
            for chunk_result in chunk_results:
                chunk_data_entry = chunk_result.copy()
                if chunk_result.get("success") and summary_idx < len(chunk_summaries):
                    chunk_data_entry["summary"] = chunk_summaries[summary_idx]
                    summary_idx += 1
                chunk_data["chunks"].append(chunk_data_entry)
            
            with open(all_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved chunk results to {all_chunks_file}")
            
            # Also save individual chunk files for easier inspection
            for i, chunk_result in enumerate(chunk_results, 1):
                if chunk_result.get("success") and i <= len(chunk_summaries):
                    chunk_file = chunk_results_dir / f"{paper_id}_chunk_{i:03d}_{timestamp}.json"
                    chunk_entry = {
                        "paper_id": paper_id,
                        "chunk_number": i,
                        "timestamp": timestamp,
                        **chunk_result,
                        "summary": chunk_summaries[i-1]
                    }
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        json.dump(chunk_entry, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"Failed to save chunk results: {e}")
            # Don't fail the whole process if saving fails

