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
   - MUST use:
       import torch_xla.core.xla_model as xm
       device = xm.xla_device()
   - MUST NOT use torch.device('cuda'), torch.device('cpu'), .to('cuda'), .cuda()
   - MUST NOT use torch_xla.device() (this API does NOT exist)

2. Optimizer:
   - MUST replace optimizer.step() → xm.optimizer_step(optimizer)
   - optimizer.zero_grad() stays the same.

3. Synchronization:
   - DO NOT introduce xm.mark_step() (deprecated)
   - DO NOT use torch_xla.sync() (THIS FUNCTION DOES NOT EXIST)
   - Manual synchronization is rarely needed.
   - XLA implicitly syncs when:
       * Calling .item()
       * Logging scalar values
   - If explicit synchronization is needed (epoch boundaries or profiling), use:
       xm.wait_device_ops()

4. Dataset:
   - load_dataset() returns EXACTLY two values:
         train_loader, test_loader
   - If code attempts:
         train_loader, val_loader, test_loader = load_dataset(...)
     → FIX to:
         train_loader, test_loader = load_dataset(...)
   - If validation is required, split train_loader or reuse test_loader.

5. Imports:
   - MUST import:
         import torch_xla.core.xla_model as xm
     for xm.xla_device(), xm.optimizer_step(), xm.wait_device_ops()
   - MUST NOT import torchvision.datasets
   - Use nn.Module, NOT xm.XlaModule

6. Tensor ops:
   - MUST use PyTorch ops (torch.matmul, torch.mm, torch.bmm, etc.)
   - MUST NOT use non-existent xm ops (xm.tensor, xm.dot_general, etc.)

7. Device placement (CRITICAL — prevents RuntimeError: bridge::IsXlaTensor):
   - ALL tensors involved in computation MUST be on XLA device:
         data = data.to(device)
         labels = labels.to(device)
         model.to(device)
   - When indexing tensors (e.g., model.alpha_bar[t]):
         t MUST ALSO be on device: t = t.to(device)
   - CPU tensor indexing XLA tensor → ALWAYS an error.

8. Neuron Core Allocation:
   - MUST NOT set NEURON_RT_NUM_CORES anywhere in code.
   - MUST remove lines such as:
         os.environ['NEURON_RT_NUM_CORES'] = '2'
   - Only 1 NeuronCore is available; keep models reasonably small.

9. HuggingFace Hub usage:
   - Datasets (IMDB, Wikitext2) come from dataset_loader (S3). DO NOT download them.
   - Pretrained models MAY be loaded if public:
         AutoTokenizer.from_pretrained(...)
         AutoModel.from_pretrained(...)
   - HTTP 401/403 → authentication required → use public model instead
   - HTTP 404 → incorrect model name
   - HTTP 429 → rate limited → retry or pick a different model

10. Model Input Shapes:
    - CNNs expect inputs shaped (batch, channels, height, width)
    - If image inputs are missing channel dim → add one via x.unsqueeze(1)
    - For 2D inputs (batch, features):
         use nn.Dropout(), NOT nn.Dropout2d()

11. Classifier Dimension Mismatch (CRITICAL):
    - Compute flattened feature size EXACTLY:
         x = features(x)
         # (optionally print shapes during debugging)
         x = x.view(x.size(0), -1)
         flattened = channels * height * width
    - Linear layer MUST match:
         nn.Linear(flattened, num_classes)
    - Common Trainium/XLA compile error:
         INVALID_ARGUMENT: Cannot infer shape for dot operation
      Almost ALWAYS means Linear input_dim does not match flattened features.

12. Transform Compatibility:
    - DO NOT use torchvision.transforms or PIL
    - Use pure PyTorch ops instead:
         x.float() / 255.0
         x.sub(mean).div(std)

13. Data Structure Validation:
    - Ensure correct assumptions:
         item['text'] only works if item is a dict
    - Use isinstance() if necessary to validate structure
═══════════════════════════════════════════════════════════════════════════════
"""
    
    prompt = f"""
You are Code Reviewer 0 — a proactive AWS Trainium compatibility reviewer.

Your task:
1. Read the ENTIRE code below.
2. Identify EVERY Trainium/XLA incompatibility.
3. Fix EVERYTHING in one pass so the code will run on Trainium without errors.
4. Preserve original logic — only fix compatibility and correctness issues.
5. Return the fully fixed code.

{paper_context}═══════════════════════════════════════════════════════════════════════════════
INITIALLY GENERATED CODE (to be reviewed for TRN compatibility):
═══════════════════════════════════════════════════════════════════════════════
```python
{code}
```

{trn_requirements}

INSTRUCTIONS:
Read the ENTIRE code carefully.

Identify ALL Trainium/XLA compatibility issues:

device handling must use xm.xla_device()
optimizer must use xm.optimizer_step()
remove xm.mark_step()
DO NOT introduce torch_xla.sync() (does not exist)
ensure all tensors & indices are moved to XLA device
confirm dataset unpacking uses exactly 2 return values
remove any NEURON_RT_NUM_CORES settings
replace Dropout2d for 2D inputs
replace torchvision transforms with PyTorch-only ops
ensure classifier Linear input_dim matches flattened CNN output
remove or fix incorrect HF Hub usage
ensure shapes match model expectations

Fix EVERYTHING in one pass.

Return output in EXACTLY this format:

```python
[FIXED CODE HERE - all Trainium/XLA compatibility fixes included, full program]
```

---FIXES_SUMMARY_START---
[Concise summary (3–5 items) of the key fixes performed]
---FIXES_SUMMARY_END---

The code block MUST come first, followed by the fixes summary in the markers.
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

