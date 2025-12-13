"""
Bedrock client for hierarchical, chunked code generation from research PDFs.
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ChunkedBedrockClient:
    def __init__(self):
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.chunk_model_id = MODEL_ID
        self.final_model_id = MODEL_ID

        from botocore.config import Config
        config = Config(
            read_timeout=150,
            retries={"max_attempts": 0},
        )
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.aws_region,
            config=config,
        )

        self.chunk_delay = float(os.getenv("CHUNK_PROCESSING_DELAY", "3.0"))

        logger.info("Chunked Bedrock client initialized")

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
        Generate an implementation-focused summary for a subset of PDF pages.
        """

        prompt = f"""
You are analyzing part {chunk_number} of {total_chunks} from a research paper.

Paper overview:
{paper_summary}

Pages covered: {page_start + 1} to {page_end}

Produce a detailed, implementation-oriented summary intended to support
a PyTorch reimplementation. Focus on:

- Mathematical definitions and equations (use clear math notation)
- Model architecture details (layers, dimensions, data flow)
- Algorithms and training procedures
- Loss functions, optimizers, schedules
- Hyperparameters and constants
- Dataset details (names, task type, data modality)
- Any explicit implementation details described in the text

Notes:
- Diagrams should be described textually if referenced
- Proof-only sections should be briefly noted but deprioritized
- Evaluation metrics may be summarized concisely

Structure the output with clear section headers.
"""

        import base64 as b64
        pdf_bytes = b64.b64decode(base64_pdf)

        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            clean_pdf_bytes = doc.tobytes(garbage=4, deflate=True)
            doc.close()
        except Exception as e:
            logger.warning(f"PDF cleanup failed, using raw bytes: {e}")
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
                                "data": pdf_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.2,
        }

        max_retries = 8
        base_delay = 5

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
                error_msg = str(e).lower()

                is_retryable = (
                    error_code in {
                        "ThrottlingException",
                        "ServiceUnavailableException",
                        "TooManyRequestsException",
                    }
                    or "throttl" in error_msg
                    or "rate" in error_msg
                )

                if is_retryable and attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), 60)
                    delay += random.uniform(0, 1)
                    logger.warning(
                        f"Chunk {chunk_number} throttled, retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                else:
                    raise

        response_body = json.loads(response["body"].read())
        summary_text = response_body["content"][0]["text"]

        return {
            "success": True,
            "summary": summary_text,
            "model_used": self.chunk_model_id,
            "pages": f"{page_start + 1}-{page_end}",
        }

    def summarize_batch(
        self,
        batch_summaries: List[str],
        batch_number: int,
        total_batches: int,
        paper_summary: str,
    ) -> Dict[str, Any]:
        """
        Consolidate a group of chunk summaries into a single intermediate summary.
        """

        combined_batch = "\n\n".join(
            f"--- Chunk {i + 1} ---\n{s}"
            for i, s in enumerate(batch_summaries)
        )

        prompt = f"""
You are consolidating summaries from batch {batch_number} of {total_batches}.

Paper overview:
{paper_summary}

Chunk summaries:
{combined_batch}

Create a concise but complete consolidated summary that:
- Removes redundancy
- Preserves all implementation-relevant details
- Keeps equations, architecture, and training logic intact

Structure the output clearly.
"""

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
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
                error_msg = str(e).lower()
                if "rate" in error_msg or "throttl" in error_msg:
                    delay = min(base_delay * (2 ** attempt), 60)
                    delay += random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise

        response_body = json.loads(response["body"].read())
        summary_text = response_body["content"][0]["text"]

        return {
            "success": True,
            "summary": summary_text,
            "model_used": self.chunk_model_id,
        }

    def generate_final_code(
        self,
        paper_summary: str,
        chunk_summaries: List[str],
        dataset_recommendations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a runnable PyTorch implementation from consolidated summaries.
        """

        combined_summaries = "\n\n".join(
            f"=== CHUNK {i + 1} ===\n{s}"
            for i, s in enumerate(chunk_summaries)
        )

        primary_dataset = (
            dataset_recommendations.get("primary_dataset", "mnist")
            if dataset_recommendations
            else "mnist"
        )

        prompt = f"""
You are generating a PyTorch implementation based on the following paper.

Paper summary:
{paper_summary}

Detailed technical summaries:
{combined_summaries}

Generate a complete, runnable PyTorch script that demonstrates the core
model, training procedure, and evaluation described in the paper.

Constraints:
- Use dataset '{primary_dataset}' via dataset_loader.load_dataset
- Target AWS Trainium using torch_xla
- Follow standard PyTorch structure
- Avoid unnecessary complexity

Output ONLY valid Python code inside a single code block.
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
                error_msg = str(e).lower()
                if "rate" in error_msg or "throttl" in error_msg:
                    delay = min(base_delay * (2 ** attempt), 120)
                    delay += random.uniform(0, 2)
                    time.sleep(delay)
                else:
                    raise

        response_body = json.loads(response["body"].read())
        generated_text = response_body["content"][0]["text"]

        code_blocks: List[str] = []
        lines = generated_text.splitlines()
        in_block = False
        buffer: List[str] = []

        for line in lines:
            if line.strip().startswith("```python"):
                in_block = True
                buffer = []
            elif line.strip().startswith("```") and in_block:
                in_block = False
                code_blocks.append("\n".join(buffer))
            elif in_block:
                buffer.append(line)

        return {
            "success": True,
            "code": "\n\n".join(code_blocks),
            "model_used": self.final_model_id,
        }