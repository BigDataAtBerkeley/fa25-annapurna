"""
Code Reviewer 0: Proactive TRN compatibility fixer.

This reviewer takes the initially generated code and fixes any issues that would
make it not compatible with TRN, BEFORE execution. This is different from other
reviewers which fix code AFTER execution failures.
"""

import os
import json
import re
import logging
import boto3
import traceback
from typing import Dict, Any, Optional
from botocore.config import Config

from utils import Config as AppConfig

logger = logging.getLogger(__name__)

bedrock_client = None
try:
    config = Config(read_timeout=150, retries={'max_attempts': 0})
    bedrock_client = boto3.client("bedrock-runtime", region_name=AppConfig.AWS_REGION, config=config)
except Exception as e:
    logger.error(f"Failed to initialize Bedrock client: {e}")


def save_code_to_s3(paper_id: str, code: str) -> Optional[str]:
    try:
        s3_client = boto3.client('s3', region_name=AppConfig.AWS_REGION)
        code_bucket = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
        s3_key = f"code/{paper_id}.py"
        
        s3_client.put_object(
            Bucket=code_bucket,
            Key=s3_key,
            Body=code.encode('utf-8'),
            ContentType='text/x-python'
        )
        logger.info(f"Saved code to S3: s3://{code_bucket}/{s3_key}")
        return s3_key
    except Exception as e:
        logger.error(f"Failed to save code to S3: {e}")
        return None


def code_reviewer_0(code: str, paper_id: str, paper_summary: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not bedrock_client:
        logger.error("Bedrock client not available for code reviewer 0")
        return None
    
    logger.info(f"Code Reviewer 0: Fixing TRN compatibility issues WITHOUT sending to trn. paper = {paper_id}")
    
    paper_context = ""
    if paper_summary:
        paper_context = f"""
PAPER CONTEXT:
{paper_summary}

"""
    
    trn_requirements = """
    TRAINIUM / XLA COMPATIBILITY RULES
    ---------------------------------
    The following constraints MUST be satisfied:

    • Device handling:
    - Use:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    - Do NOT use CUDA, CPU-only device strings, or torch_xla.device()

    • Optimizer:
    - Replace optimizer.step() with xm.optimizer_step(optimizer)
    - optimizer.zero_grad() remains unchanged

    • Synchronization:
    - Do NOT use xm.mark_step()
    - Do NOT use torch_xla.sync()
    - If explicit sync is required, use xm.wait_device_ops()

    • Dataset loading:
    - load_dataset() returns EXACTLY:
            train_loader, test_loader
    - Do NOT unpack validation loaders directly

    • Imports and APIs:
    - Do NOT import torchvision.datasets or torchvision.transforms
    - Do NOT use nonexistent xm.* tensor ops

    • Device placement:
    - ALL tensors, models, and indices must be moved to XLA device
    - Indexing an XLA tensor with a CPU tensor is invalid

    • Model correctness:
    - CNN classifier input dimensions must match flattened feature size
    - Use Dropout for 2D inputs, NOT Dropout2d

    • Environment:
    - Do NOT set NEURON_RT_NUM_CORES
    - Assume a single NeuronCore

    • HuggingFace usage:
    - Only public models allowed
    - Dataset downloads must come from dataset_loader, not HF Datasets
    """
    
    prompt = f"""
You are Code Reviewer 0 — a proactive AWS Trainium compatibility reviewer.

Task: Fix all Trainium/XLA compatibility issues in the code below while preserving original logic.

{paper_context}CODE TO REVIEW:
```python
{code}
```

{trn_requirements}

Return output in this format:

```python
[FIXED CODE HERE]
```

---FIXES_SUMMARY_START---
[Summary of fixes]
---FIXES_SUMMARY_END---
"""
    
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0.2, 
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = bedrock_client.invoke_model(
            modelId=AppConfig.BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        generated_text = response_body['content'][0]['text']
        
        fixed_code = None
        fixes_summary = ["TRN compatibility fixes applied via Code Reviewer 0"] 
        
        code_match = re.search(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
        if code_match:
            fixed_code = code_match.group(1).strip()
        else:
            text_before_summary = generated_text.split('---FIXES_SUMMARY_START---')[0].strip()
            if text_before_summary.startswith('import') or text_before_summary.startswith('from'):
                fixed_code = text_before_summary
        
        summary_match = re.search(r'---FIXES_SUMMARY_START---\s*(.*?)\s*---FIXES_SUMMARY_END---', generated_text, re.DOTALL)
        if summary_match:
            summary_text = summary_match.group(1).strip()
            if summary_text:
                summary_lines = [line.strip().lstrip('- ').strip() for line in summary_text.split('\n') if line.strip()]
                if summary_lines:
                    fixes_summary = summary_lines
                else:
                    fixes_summary = [summary_text]
        
        if fixed_code:
            code_changed = fixed_code != code
            save_code_to_s3(paper_id, fixed_code)
            return {
                "code": fixed_code,
                "fixes_summary": fixes_summary,
                "code_changed": code_changed
            }
        
        logger.warning("Code Reviewer 0: Failed to extract code from Bedrock response")
        return None
        
    except Exception as e:
        logger.error(f"Code Reviewer 0: Failed to fix code with Bedrock: {e}")
        logger.error(traceback.format_exc())
        return None

