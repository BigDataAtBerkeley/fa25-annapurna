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
from botocore.exceptions import ClientError

from utils import Config as AppConfig

logger = logging.getLogger(__name__)

# Initialize Bedrock client
bedrock_client = None
try:
    config = Config(read_timeout=150, retries={'max_attempts': 0})
    bedrock_client = boto3.client("bedrock-runtime", region_name=AppConfig.AWS_REGION, config=config)
    logger.info(f"Bedrock client initialized for code reviewer 0: {AppConfig.BEDROCK_MODEL_ID}")
except Exception as e:
    logger.error(f"Failed to initialize Bedrock client: {e}")


def save_code_to_s3(paper_id: str, code: str) -> Optional[str]:
    """
    Save code to S3.
    
    Args:
        paper_id: Paper ID
        code: Code content
        
    Returns:
        S3 key if successful, None otherwise
    """
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
    """
    Code Reviewer 0: Proactive TRN compatibility fixer.
    
    This reviewer takes the initially generated code and fixes any issues that would
    make it not compatible with TRN, BEFORE execution. This is different from other
    reviewers which fix code AFTER execution failures.
    
    Args:
        code: Initially generated code
        paper_id: Paper ID (for logging)
        paper_summary: Optional paper summary for context
        
    Returns:
        Dictionary with 'code' and 'fixes_summary' keys, or None if fixing failed
    """
    if not bedrock_client:
        logger.error("Bedrock client not available for code reviewer 0")
        return None
    
    logger.info(f"Code Reviewer 0: Fixing TRN compatibility issues WITHOUT sending to trn. paper = {paper_id}")
    
    # Pre-validate code for common issues
    detected_issues = []
    if re.search(r'(\w+),\s*(\w+),\s*(\w+)\s*=\s*load_dataset', code):
        detected_issues.append("3-value unpacking from load_dataset() detected")
        logger.warning(f"Code Reviewer 0: Detected 3-value unpacking pattern - will be fixed")
    
    if re.search(r'xm\.xla_device\(\)', code):
        detected_issues.append("Deprecated xm.xla_device() usage detected")
        logger.warning(f"Code Reviewer 0: Detected deprecated xm.xla_device() - will be fixed")
    
    if re.search(r'NEURON_RT_NUM_CORES\s*=', code, re.IGNORECASE):
        detected_issues.append("NEURON_RT_NUM_CORES setting detected")
        logger.warning(f"Code Reviewer 0: Detected NEURON_RT_NUM_CORES setting - will be removed")
    
    if re.search(r'os\.environ\[.NEURON_RT_NUM_CORES', code, re.IGNORECASE):
        detected_issues.append("NEURON_RT_NUM_CORES environment variable setting detected")
        logger.warning(f"Code Reviewer 0: Detected NEURON_RT_NUM_CORES env var setting - will be removed")
    
    # Note: .from_pretrained() is OK for models (fine-tuning), but not for datasets
    # We'll let Code Reviewer 0 handle this case-by-case based on context
    
    paper_context = ""
    if paper_summary:
        paper_context = f"""
PAPER CONTEXT (for better understanding of what the code should implement):
{paper_summary}

"""
    
    # Essential TRN/XLA requirements guidance
    trn_requirements = """
═══════════════════════════════════════════════════════════════════════════════
ESSENTIAL TRAINIUM/XLA REQUIREMENTS (verify and fix all of these):
═══════════════════════════════════════════════════════════════════════════════
1. Device handling:
   - MUST use: device = torch_xla.device() (NOT xm.xla_device() - that's deprecated)
   - MUST NOT use: torch.device('cuda'), torch.device('cpu'), .to('cuda'), .cuda()

2. Optimizer:
   - MUST replace optimizer.step() → xm.optimizer_step(optimizer)

3. Synchronization:
   - MUST call xm.mark_step() after each backward() + optimizer step
   - MUST NOT call .item() inside training loop except for logging

4. Dataset:
   - load_dataset() returns EXACTLY 2 DataLoaders: (train_loader, test_loader)
   - CRITICAL: If code tries to unpack 3 values like "train_loader, val_loader, test_loader = load_dataset(...)", 
     fix it to unpack only 2: "train_loader, test_loader = load_dataset(...)"
   - MUST NOT treat them as raw datasets
   - If validation is needed, split train_loader or use test_loader for validation

5. Imports:
   - MUST import: import torch_xla (for device) and import torch_xla.core.xla_model as xm (for optimizer_step, mark_step)
   - MUST NOT import torchvision datasets
   - Use nn.Module, NOT xm.XlaModule

6. Tensor ops:
   - MUST use torch.matmul / torch.mm etc.
   - MUST NOT use xm-specific ops (xm.tensor, xm.dot_general, etc.)

7. Device placement:
   - All model, input, labels, and loss tensors MUST be moved to device

8. Neuron Core Allocation:
   - MUST NOT set NEURON_RT_NUM_CORES environment variable
   - MUST NOT use os.environ['NEURON_RT_NUM_CORES'] = '2' or any value
   - MUST use default single-core allocation (only 1 core available on instance)
   - If code sets NEURON_RT_NUM_CORES, remove those lines completely
   - Keep models small enough to fit in 1 core (avoid very large models that require 2+ cores)
   - If model compilation requires 2 cores, reduce model size or complexity

9. HuggingFace Hub Usage:
   - Datasets: dataset_loader provides wikitext/imdb from S3 - NO dataset downloads from HF Hub needed
   - Models for fine-tuning: CAN use AutoModel.from_pretrained() or AutoTokenizer.from_pretrained()
     * Use publicly available models (e.g., 'bert-base-uncased', 'gpt2')
     * HUGGINGFACE_HUB_TOKEN may be available if needed
     * If HTTP errors occur, try a different public model or check if token is needed
   - If error shows HTTP 401/403: model requires authentication - use a public model instead
   - If error shows HTTP 404: model name is wrong - fix the model name
═══════════════════════════════════════════════════════════════════════════════
"""
    
    prompt = f"""
You are Code Reviewer 0 — a proactive AWS Trainium compatibility reviewer.

Your task:
1. Read the ENTIRE code below.
2. Identify EVERY TRN/XLA incompatibility.
3. Fix EVERYTHING in *one pass* so the code will run on Trainium without errors.
4. Preserve original logic — only modify compatibility-related issues.
5. Return the fixed code in full.

{paper_context}═══════════════════════════════════════════════════════════════════════════════
INITIALLY GENERATED CODE (to be reviewed for TRN compatibility):
═══════════════════════════════════════════════════════════════════════════════
```python
{code}
```

{trn_requirements}

INSTRUCTIONS:
1. Read through the ENTIRE code carefully
2. Identify ALL TRN/XLA compatibility issues (device, optimizer, synchronization, datasets, imports)
3. Check for runtime errors that would cause immediate failures:
   - Unpacking errors: Verify number of variables matches function return values
     * CRITICAL: load_dataset() returns exactly 2 values (train_loader, test_loader)
     * Fix any 3-value unpacking: "train_loader, val_loader, test_loader = load_dataset(...)" → "train_loader, test_loader = load_dataset(...)"
   - Neuron Core Allocation errors: Remove ALL lines that set NEURON_RT_NUM_CORES
     * CRITICAL: If error shows "Logical Neuron Core(s) not available - Requested:2 Available:0"
     * Remove lines like: os.environ['NEURON_RT_NUM_CORES'] = '2' or NEURON_RT_NUM_CORES = 2
     * Only 1 core is available - code must use default single-core allocation
   - HuggingFace Hub errors: Fix based on error type
     * HTTP 401/403: Model requires authentication - change to a public model (e.g., 'bert-base-uncased')
     * HTTP 404: Model name is wrong - fix the model name
     * HTTP 429: Rate limited - add retry logic or use a different model
     * For datasets: dataset_loader provides wikitext/imdb from S3 - don't download datasets from HF Hub
   - Data structure errors: Verify data access patterns match actual data types
     * If accessing item['key'], ensure item is a dict, not a list
     * Add isinstance() checks if data format is uncertain
   - API contract mismatches: Ensure function calls match their return signatures
   - Common errors: ValueError, TypeError, AttributeError that would occur on first execution
4. Fix ALL issues in one pass - don't leave any compatibility problems
5. Ensure the code will work on Trainium without execution errors
6. Preserve the original code logic and functionality - only fix compatibility and correctness issues

CRITICAL: You must return BOTH the fixed code AND a summary of fixes in the following exact format:

```python
[FIXED CODE HERE - complete Python code with all TRN compatibility fixes]
```

---FIXES_SUMMARY_START---
[Summary of TRN compatibility fixes made - concise list of what was changed]
---FIXES_SUMMARY_END---

The fixes summary should be a brief list (3-5 items max) of the key TRN compatibility changes made. Example:
- Changed device from torch.device('cuda') or xm.xla_device() to torch_xla.device()
- Replaced optimizer.step() with xm.optimizer_step(optimizer)
- Added xm.mark_step() after backward pass
- Fixed dataset unpacking: changed "train_loader, val_loader, test_loader = load_dataset(...)" to "train_loader, test_loader = load_dataset(...)" (load_dataset returns 2 values, not 3)

IMPORTANT: The code block must come FIRST, followed by the fixes summary between the markers.
"""
    
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 8192,
            "temperature": 0.2,  # Lower temperature for more consistent fixes
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
        
        # Extract code and summary separately
        fixed_code = None
        fixes_summary = ["TRN compatibility fixes applied via Code Reviewer 0"]  # Default
        
        # Extract code block (must come first)
        code_match = re.search(r'```python\n(.*?)\n```', generated_text, re.DOTALL)
        if code_match:
            fixed_code = code_match.group(1).strip()
        else:
            # Fallback: try to extract code without markdown
            text_before_summary = generated_text.split('---FIXES_SUMMARY_START---')[0].strip()
            if text_before_summary.startswith('import') or text_before_summary.startswith('from'):
                fixed_code = text_before_summary
        
        # Extract fixes summary (between markers)
        summary_match = re.search(r'---FIXES_SUMMARY_START---\s*(.*?)\s*---FIXES_SUMMARY_END---', generated_text, re.DOTALL)
        if summary_match:
            summary_text = summary_match.group(1).strip()
            if summary_text:
                # Parse summary into list if it's bullet points or lines
                summary_lines = [line.strip().lstrip('- ').strip() for line in summary_text.split('\n') if line.strip()]
                if summary_lines:
                    fixes_summary = summary_lines
                else:
                    fixes_summary = [summary_text]
        
        if fixed_code:
            # Log fixes made
            code_changed = fixed_code != code
            if code_changed:
                logger.info(f"✅ Code Reviewer 0: Fixed code extracted (length: {len(fixed_code)} chars)")
                logger.info(f"   Code length: {len(code)} → {len(fixed_code)} chars")
                if detected_issues:
                    logger.info(f"   Pre-detected issues: {', '.join(detected_issues)}")
                summary_str = ', '.join(fixes_summary) if isinstance(fixes_summary, list) else str(fixes_summary)
                logger.info(f"   Fixes summary: {summary_str[:200]}")
                
                # Save fixed code to S3 (replaces old code)
                s3_key = save_code_to_s3(paper_id, fixed_code)
                if s3_key:
                    logger.info(f"✅ Code Reviewer 0: Saved fixed code to S3: {s3_key}")
                else:
                    logger.warning(f"⚠️ Code Reviewer 0: Failed to save fixed code to S3")
            else:
                logger.info(f"ℹ️ Code Reviewer 0: No changes needed - code is already TRN compatible")
                # Still save to S3 even if unchanged (ensures code is persisted)
                s3_key = save_code_to_s3(paper_id, fixed_code)
                if s3_key:
                    logger.info(f"✅ Code Reviewer 0: Saved code to S3 (unchanged): {s3_key}")
            
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

