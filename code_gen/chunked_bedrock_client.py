"""
Bedrock client for chunked code generation.
Handles individual chunk summarization and final code generation.

"""

import os
import json
import time
import random
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

load_dotenv()

import logging

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)




class ChunkedBedrockClient:
    """Client for chunked code generation with throttling mitigation."""

    def __init__(self):
        """
        Initialize Bedrock client for chunked generation.
        """
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.chunk_model_id = MODEL_ID
        self.final_model_id = MODEL_ID
        self.model_id = MODEL_ID

        from botocore.config import Config

        config = Config(
            read_timeout=150,
            retries={"max_attempts": 0},
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.aws_region, config=config)

        self.chunk_delay = float(os.getenv("CHUNK_PROCESSING_DELAY", "3.0"))  # 3 seconds between chunk processing

        logger.info("Chunked Bedrock client initialized:")
        logger.info(f"  - Chunk model: {self.chunk_model_id}")
        logger.info(f"  - Final model: {self.final_model_id}")
        logger.info(f"  - Chunk delay: {self.chunk_delay}s")

    def summarize_pdf_chunk(
        self,
        base64_pdf: str,
        chunk_number: int,
        total_chunks: int,
        paper_summary: str,
        page_start: int,
        page_end: int,
    ) -> Dict[str, Any]:
        """
        Generate a detailed summary of a PDF chunk using PDF document input.
        Extracts formulas and diagrams from PDF pages.

        Args:
            base64_pdf: Base64-encoded PDF string (containing the relevant pages)
            chunk_number: Current chunk number (1-indexed)
            total_chunks: Total number of chunks
            paper_summary: Overall paper summary (title, authors, abstract)
            page_start: Starting page number (0-indexed, for display)
            page_end: Ending page number (exclusive, 0-indexed, for display)

        Returns:
            Dictionary with summary and metadata
        """
        prompt = f"""
You are analyzing a research paper PDF that has been split into {total_chunks} parts. You are analyzing part {chunk_number} of {total_chunks}, which contains pages {page_start + 1} through {page_end}.

Paper Overview:
{paper_summary}

Your task is to analyze the provided PDF pages and provide a DETAILED SUMMARY that will be used to generate PyTorch code. Focus on extracting:

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

        import base64 as b64
        pdf_bytes = b64.b64decode(base64_pdf)
        
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            clean_pdf_bytes = doc.tobytes(garbage=4, deflate=True)
            doc.close()
        except Exception as e:
            logger.warning(f"Could not clean PDF, using original: {e}")
            clean_pdf_bytes = pdf_bytes
        
        pdf_b64 = b64.b64encode(clean_pdf_bytes).decode("utf-8")
        
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
            "max_tokens": 4096,
            "temperature": 0.2
        }

        max_retries = 8
        base_delay = 5

        for attempt in range(max_retries):
            try:
                # Messages API 
                response = self.client.invoke_model(
                    modelId=self.chunk_model_id,
                    body=json.dumps(body),
                    contentType="application/json"
                )
                break
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_msg = str(e)

                logger.debug(
                    f"PDF Chunk {chunk_number} error: Code={error_code}, Message={error_msg[:200]}"
                )

                retryable_errors = [
                    "ThrottlingException",
                    "ServiceUnavailableException",
                    "TooManyRequestsException",
                ]
                is_throttling = (
                    error_code in retryable_errors
                    or "throttl" in error_msg.lower()
                    or "too many requests" in error_msg.lower()
                    or "rate limit" in error_msg.lower()
                    or "rate exceeded" in error_msg.lower()
                )

                if is_throttling and attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), 60)  # Cap at 60s for chunks
                    jitter = random.uniform(0, 1)
                    total_delay = delay + jitter
                    logger.warning(
                        f"PDF Chunk {chunk_number} throttling detected (Code: {error_code}), "
                        f"retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    if not is_throttling:
                        logger.error(
                            f"PDF Chunk {chunk_number} non-retryable error: "
                            f"Code={error_code}, Message={error_msg[:200]}"
                        )
                    raise

        # Extract text from Messages API response
        response_body = json.loads(response["body"].read())
        summary_text = response_body["content"][0]["text"]

        return {
            "success": True,
            "chunk_number": chunk_number,
            "summary": summary_text,
            "model_used": self.chunk_model_id,
            "pages": f"{page_start + 1}-{page_end}",
            "num_pages": page_end - page_start,
        }

    def summarize_batch(
        self,
        batch_summaries: List[str],
        batch_number: int,
        total_batches: int,
        paper_summary: str,
    ) -> Dict[str, Any]:
        """
        Summarize a batch of chunk summaries into an intermediate summary (Stage 2 of hierarchical summarization).

        Args:
            batch_summaries: List of summaries from a batch of chunks
            batch_number: Current batch number (1-indexed)
            total_batches: Total number of batches
            paper_summary: Overall paper summary

        Returns:
            Dictionary with intermediate summary and metadata
        """
        # Combine batch summaries
        combined_batch = "\n\n".join(
            [f"--- Chunk Summary {i+1} ---\n{summary}" for i, summary in enumerate(batch_summaries)]
        )

        prompt = f"""
You are analyzing intermediate summaries from a research paper. These summaries come from batch {batch_number} of {total_batches} batches.

Paper Overview:
{paper_summary}

Batch {batch_number} Summaries:
{combined_batch}

Your task is to create a CONSOLIDATED SUMMARY of this batch that:
1. Combines and synthesizes the key information from all chunk summaries in this batch
2. Removes redundancy and focuses on unique, important details
3. Preserves all mathematical formulas, algorithms, and implementation details
4. Maintains architectural and training procedure information
5. Is comprehensive but concise

This consolidated summary will be combined with other batch summaries to generate final PyTorch code.

Format your response as a structured, comprehensive summary.
"""

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,  # Increased from 4096 to handle consolidating 8 chunk summaries
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        max_retries = 8
        base_delay = 3

        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.chunk_model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                break
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_msg = str(e)

                retryable_errors = [
                    "ThrottlingException",
                    "ServiceUnavailableException",
                    "TooManyRequestsException",
                ]
                is_throttling = (
                    error_code in retryable_errors
                    or "throttl" in error_msg.lower()
                    or "too many requests" in error_msg.lower()
                    or "rate" in error_msg.lower()
                )

                if is_throttling and attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), 60)
                    jitter = random.uniform(0, 1)
                    total_delay = delay + jitter
                    logger.warning(
                        f"Batch {batch_number} throttling detected, "
                        f"retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    raise

        response_body = json.loads(response["body"].read())
        summary_text = response_body["content"][0]["text"]

        # Check for truncation
        stop_reason = response_body.get("stop_reason")
        truncated = stop_reason == "max_tokens"
        if truncated:
            logger.warning(
                f"⚠️ Batch {batch_number} summary was truncated due to max_tokens limit (8192). "
                "Some details may be missing."
            )

        return {
            "success": True,
            "batch_number": batch_number,
            "summary": summary_text,
            "model_used": self.chunk_model_id,
            "truncated": truncated,
        }

    def generate_final_code(
        self,
        paper_summary: str,
        chunk_summaries: List[str],
        dataset_recommendations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate final PyTorch code from combined chunk summaries.

        Args:
            paper_summary: Overall paper summary
            chunk_summaries: List of detailed summaries from each chunk
            dataset_recommendations: Optional dataset recommendations

        Returns:
            Dictionary with generated code and metadata
        """
        # Combine all chunk summaries
        combined_summaries = "\n\n".join(
            [f"=== CHUNK {i+1} SUMMARY ===\n{summary}" for i, summary in enumerate(chunk_summaries)]
        )

        # Get primary dataset
        primary_dataset = "mnist"
        if dataset_recommendations:
            primary_dataset = dataset_recommendations.get("primary_dataset", "mnist")

        prompt = f"""
You are an expert PyTorch developer and machine learning researcher. I will provide you with detailed summaries from different parts of a research paper, and I need you to generate a complete PyTorch implementation that demonstrates the key concepts and algorithms described in the paper.

Paper Information:
{paper_summary}

Detailed Chunk Summaries:
{combined_summaries}

"""

        # Add dataset recommendations if available
        if dataset_recommendations:
            recommended_datasets = dataset_recommendations.get("recommended_datasets", [])
            domain = dataset_recommendations.get("domain", "general")
            explicitly_mentioned = dataset_recommendations.get("explicitly_mentioned", [])
            reasoning = dataset_recommendations.get("llm_reasoning") or dataset_recommendations.get(
                "reasoning", ""
            )

            prompt += f"""
DATASET RECOMMENDATIONS (IMPORTANT - USE THESE):
PRIMARY DATASET: {primary_dataset}
RECOMMENDED DATASETS: {', '.join(recommended_datasets) if recommended_datasets else 'synthetic'}
INFERRED DOMAIN: {domain}
"""
            if explicitly_mentioned:
                prompt += f"EXPLICITLY MENTIONED IN PAPER: {', '.join(explicitly_mentioned)}\n"
            if reasoning:
                prompt += f"REASONING: {reasoning}\n"
            prompt += "\n"

        prompt += f"""
Generate a complete, production-ready PyTorch implementation that demonstrates the paper's key concepts.

CRITICAL REQUIREMENTS:
1. Use dataset '{primary_dataset}' via: `from dataset_loader import load_dataset` (DO NOT use torchvision.datasets)
   - CRITICAL: load_dataset() returns EXACTLY 2 DataLoaders: (train_loader, test_loader)
   - NEVER unpack 3 values like "train_loader, val_loader, test_loader = load_dataset(...)"
   - If validation is needed, split train_loader or use test_loader for validation
   - IMPORTANT: load_dataset() automatically downloads the dataset from S3 if not already cached - no manual download needed
   - Simply call: train_loader, test_loader = load_dataset('{primary_dataset}', batch_size=128)
   
   DATASET FORMATS (CRITICAL - understand what each dataset returns):
   - Image datasets (mnist, cifar10, cifar100, fashion_mnist, synthetic): 
     * Returns (image_tensor, label_tensor) where labels are SCALAR integers (0, 1, 2, ...)
     * Image shape: [batch, channels, height, width] or [batch, height, width] for grayscale
     * Label shape: [batch] - each label is a single integer class ID
   - Text classification (imdb):
     * Returns (text_string, label_tensor) where labels are SCALAR integers (0 or 1)
     * Text: Python strings (need tokenization in your code)
     * Label shape: [batch] - each label is a single integer (0=negative, 1=positive)
   - Language modeling (wikitext2):
     * Returns (input_ids_tensor, labels_tensor) where BOTH are SEQUENCES of token IDs
     * Input shape: [batch, seq_length] - token IDs for input sequence
     * Label shape: [batch, seq_length] - token IDs for next-token prediction (shifted by 1)
     * CRITICAL: Labels are NOT scalars - they are sequences! Use them directly in loss functions like CrossEntropyLoss
     * Example: loss = criterion(logits.view(-1, vocab_size), labels.view(-1))  # Flatten for loss calculation
2. Use Trainium/XLA: `import torch_xla` and `device = torch_xla.device()` (NOT xm.xla_device())
3. Use `xm.optimizer_step(optimizer)` instead of `optimizer.step()` and call `xm.mark_step()` after
4. Import ALL modules you use (math, random, collections, etc.)

═══════════════════════════════════════════════════════════════════════════════
PACKAGES & ENVIRONMENT
═══════════════════════════════════════════════════════════════════════════════

AVAILABLE (Neuron SDK Environment):
- torch, torch_xla (Neuron SDK PyTorch 2.1.0 with XLA support) - REQUIRED for Trainium
- torch.nn, torch.optim (PyTorch neural network and optimizer modules)
- numpy, standard library (math, random, collections, json, os, sys, etc.)
- dataset_loader (custom module) - provides pre-processed datasets with tokenization already done

HuggingFace Hub Usage:
- Datasets (wikitext/imdb): dataset_loader already provides pre-processed data from S3 - NO downloads needed
- Models/Tokenizers for fine-tuning: Can use AutoTokenizer.from_pretrained() or AutoModel.from_pretrained() IF:
  * The model is publicly available (no private repo)
  * HUGGINGFACE_HUB_TOKEN may be set (optional, only needed for private models)
  * For fine-tuning papers: you can load pre-trained models from HuggingFace Hub
- If you get HTTP errors: the model may require authentication or the instance has no internet access

NOTE: This code runs on AWS Trainium using the Neuron SDK. The torch_xla module is part of the Neuron SDK and provides XLA (Accelerated Linear Algebra) support for Trainium accelerators.

IMPORTANT RESOURCE CONSTRAINTS:
- Do NOT set NEURON_RT_NUM_CORES or request multiple cores - use default single-core allocation
- Only 1 Neuron core is available - keep models small enough to fit in 1 core
- Large models (e.g., very large transformers) may require 2+ cores and will fail
- Initialize device BEFORE moving models to device: device = torch_xla.device() must come first
- Only one execution runs at a time on the instance

NOT AVAILABLE / DO NOT USE:
- transformers_xla (THIS PACKAGE DOES NOT EXIST)
- XLATokenizer (THIS CLASS DOES NOT EXIST)
- matplotlib, PIL/Pillow, pandas, scipy, sklearn, torchtext
- torchvision.datasets (use dataset_loader instead)

NOTE on HuggingFace Hub:
- Datasets: dataset_loader provides wikitext/imdb from S3 - no HF Hub download needed
- Models: For fine-tuning, you CAN use AutoModel.from_pretrained() or AutoTokenizer.from_pretrained()
  * Use publicly available models (e.g., 'bert-base-uncased', 'gpt2', 'distilbert-base-uncased')
  * HUGGINGFACE_HUB_TOKEN may be available for private models
  * If HTTP errors occur, the model may require authentication or instance lacks internet access

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT - CODE WITH BRIEF INLINE COMMENTS
═══════════════════════════════════════════════════════════════════════════════

CRITICAL: You have a 8192 token OUTPUT limit. Generate complete, detailed code with brief inline comments explaining key implementation steps.

Your response must be EXACTLY this format:

```python
[COMPLETE, RUNNABLE PYTORCH CODE]
# All imports here
# Model definition with brief comments explaining key components
# Dataset loading
# Training loop with comments for important steps
# Evaluation
# Everything needed to run
```

INCLUDE:
- Complete, detailed implementation based on ALL chunk summaries
- Brief inline comments explaining key steps
- All necessary code to run the implementation

DO NOT include:
- Long explanations before or after the code block
- Markdown headers or sections outside code
- "Here's the code" or similar introductory text

ONLY output the code block with inline comments. The code must be immediately runnable with no manual setup required.
"""

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        max_retries = 8
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.final_model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                break
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_msg = str(e)

                retryable_errors = [
                    "ThrottlingException",
                    "ServiceUnavailableException",
                    "TooManyRequestsException",
                ]
                is_throttling = (
                    error_code in retryable_errors
                    or "throttl" in error_msg.lower()
                    or "too many requests" in error_msg.lower()
                    or "rate" in error_msg.lower()
                )

                if is_throttling and attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), 120)
                    jitter = random.uniform(0, 2)
                    total_delay = delay + jitter
                    logger.warning(
                        "Final code generation throttling detected, "
                        f"retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    raise

        response_body = json.loads(response["body"].read())
        generated_text = response_body["content"][0]["text"]

        # Extract code from response
        code_blocks: List[str] = []
        lines = generated_text.split("\n")
        in_code_block = False
        current_code: List[str] = []

        for line in lines:
            if line.strip().startswith("```python"):
                in_code_block = True
                current_code = []
                continue
            elif line.strip().startswith("```") and in_code_block:
                in_code_block = False
                if current_code:
                    code_blocks.append("\n".join(current_code))
                current_code = []
                continue
            elif in_code_block:
                current_code.append(line)

        full_code = "\n\n".join(code_blocks) if code_blocks else ""

        return {
            "success": True,
            "code": full_code,
            "full_response": generated_text,
            "model_used": self.final_model_id,
        }



