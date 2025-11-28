"""
AWS Bedrock client for interacting with Claude to generate PyTorch code.
"""

import os
import json
import logging
import time
import random
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class BedrockClient:
    """Client for interacting with AWS Bedrock Claude models."""
    
    def __init__(self):
        """Initialize  Bedrock client"""
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        # Per-paper timeout is 180s, so we set Bedrock timeout to 150s to ensure
        # Bedrock calls timeout before the per-paper timeout fires
        from botocore.config import Config
        config = Config(
            read_timeout=150,  # 2.5 minutes - must be less than per-paper timeout (180s)
            retries={'max_attempts': 0}  
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.aws_region, config=config)
        
        logger.info(f"Bedrock client initialized with model: {self.model_id} (timeout: 150s)")
    
    def generate_pytorch_code(self, paper_summary: str, paper_content: Optional[str] = None,
                             dataset_recommendations: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate PyTorch code based on a research paper.
        
        Args:
            paper_summary: Summary of the paper (title, authors, abstract)
            paper_content: Full paper content (optional)
            dataset_recommendations: Optional dataset recommendations dictionary
            
        Returns:
            Dictionary containing the generated code and metadata
        """
        try:
            # Prepare the prompt for Claude
            prompt = self._create_pytorch_prompt(paper_summary, paper_content, dataset_recommendations)
            
            # Prepare the request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,  # Claude 3.5 Sonnet supports up to 8,192 tokens output
                "temperature": 0.3,  # Lower temperature for more detailed, deterministic code generation
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            max_retries = 8  # Increased from 5 to handle rate limits better
            base_delay = 5  # Increased from 2s to 5s for better rate limit handling
            for attempt in range(max_retries):
                try:
                    response = self.client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(body),
                        contentType="application/json"
                    )
                    break  # Success, exit retry loop
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    error_msg = str(e)
                    # Retry on throttling or service unavailability (both are often transient)
                    retryable_errors = ['ThrottlingException', 'ServiceUnavailableException', 'TooManyRequestsException']
                    # Also check error message for throttling indicators
                    is_throttling = (error_code in retryable_errors or 
                                   'throttl' in error_msg.lower() or 
                                   'too many requests' in error_msg.lower() or
                                   'rate' in error_msg.lower())
                    
                    if is_throttling and attempt < max_retries - 1:
                        # Exponential backoff with jitter: 5s, 10s, 20s, 40s, 80s, 120s, 120s, 120s
                        # Cap at 120s to avoid extremely long waits
                        delay = min(base_delay * (2 ** attempt), 120)
                        # Add random jitter (0-2s) to avoid thundering herd
                        jitter = random.uniform(0, 2)
                        total_delay = delay + jitter
                        logger.warning(f"Bedrock throttling detected ({error_code}), retrying in {total_delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(total_delay)
                        continue
                    else:
                        raise  # Re-raise if not retryable or out of retries
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            generated_text = response_body['content'][0]['text']
            
            # Check if response was truncated due to max_tokens
            stop_reason = response_body.get('stop_reason')
            if stop_reason == 'max_tokens':
                logger.warning("⚠️ Bedrock response was truncated due to max_tokens limit (8192). Code may be incomplete!")
            
            # Extract code from the response
            code_result = self._extract_code_from_response(generated_text)
            
            # Check if extracted code appears incomplete
            code = code_result.get('code', '')
            if code and stop_reason == 'max_tokens':
                # Check for incomplete code patterns
                incomplete_patterns = [
                    code.rstrip().endswith('#'),  # Ends with comment
                    code.rstrip().endswith(','),  # Ends with comma
                    code.rstrip().endswith('('),  # Ends with open paren
                    code.rstrip().endswith('['),  # Ends with open bracket
                    code.rstrip().endswith('{'),  # Ends with open brace
                    not code.rstrip().endswith('\n'),  # Doesn't end with newline (might be cut off)
                ]
                if any(incomplete_patterns):
                    logger.warning("⚠️ Extracted code appears incomplete - may be truncated")
                    code_result['truncated'] = True
                    code_result['warning'] = "Code generation was truncated due to max_tokens limit. Code may be incomplete."
            
            logger.info("Successfully generated PyTorch code")
            result = {
                "success": True,
                "code": code_result["code"],
                "explanation": code_result["explanation"],
                "full_response": generated_text,
                "model_used": self.model_id
            }
            
            # Add truncation warning if detected
            if code_result.get('truncated'):
                result["truncated"] = True
                result["warning"] = code_result.get('warning', 'Code may be incomplete due to max_tokens limit')
            if stop_reason == 'max_tokens':
                result["stop_reason"] = "max_tokens"
                result["truncated"] = True
                if "warning" not in result:
                    result["warning"] = "Code generation was truncated due to max_tokens limit (8192). Code may be incomplete."
            
            return result
            
        except ClientError as e:
            logger.error(f"Bedrock client error: {e}")
            return {
                "success": False,
                "error": f"Bedrock error: {str(e)}",
                "code": None,
                "explanation": None
            }
        except Exception as e:
            logger.error(f"Unexpected error generating code: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "code": None,
                "explanation": None
            }
    
    def _create_pytorch_prompt(self, paper_summary: str, paper_content: Optional[str] = None,
                               dataset_recommendations: Optional[Dict[str, Any]] = None) -> str:
        """
        ACTUAL PROMPT SENT TO BEDROCK TO GENERATE CODE
        
        Args:
            paper_summary: Summary of the paper
            paper_content: Full paper content (optional)
            dataset_recommendations: Optional dataset recommendations dictionary
            
        Returns:
            Formatted prompt string
        """
        base_prompt = f"""
You are an expert PyTorch developer and machine learning researcher. I will provide you with information about a research paper, and I need you to generate a complete PyTorch implementation that demonstrates the key concepts and algorithms described in the paper.

Paper Information:
{paper_summary}

"""

        if paper_content:
            base_prompt += f"""
Full Paper Content:
{paper_content[:150000]}

"""

        # Add dataset recommendations if available
        if dataset_recommendations:
            recommended_datasets = dataset_recommendations.get("recommended_datasets", [])
            primary_dataset = dataset_recommendations.get("primary_dataset", "synthetic")
            domain = dataset_recommendations.get("domain", "general")
            explicitly_mentioned = dataset_recommendations.get("explicitly_mentioned", [])
            reasoning = dataset_recommendations.get("llm_reasoning") or dataset_recommendations.get("reasoning", "")
            
            base_prompt += f"""
DATASET RECOMMENDATIONS (IMPORTANT - USE THESE):
Based on analysis of this paper, the following datasets are recommended:

PRIMARY DATASET: {primary_dataset}
RECOMMENDED DATASETS: {', '.join(recommended_datasets) if recommended_datasets else 'synthetic'}
INFERRED DOMAIN: {domain}
"""
            
            if explicitly_mentioned:
                base_prompt += f"EXPLICITLY MENTIONED IN PAPER: {', '.join(explicitly_mentioned)}\n"
            
            if reasoning:
                base_prompt += f"REASONING: {reasoning}\n"
            
            base_prompt += "\n"

        # Get primary dataset name for use in prompt
        primary_dataset = "mnist"  # Default fallback
        if dataset_recommendations:
            primary_dataset = dataset_recommendations.get("primary_dataset", "mnist")

        base_prompt += f"""
Generate a complete, production-ready PyTorch implementation that demonstrates the paper's key concepts.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS - READ CAREFULLY
═══════════════════════════════════════════════════════════════════════════════

1. DATASET LOADING (MANDATORY):
   ```python
   from dataset_loader import load_dataset
   train_loader, test_loader = load_dataset('{primary_dataset}', batch_size=128)
   ```
   - USE PRIMARY DATASET: '{primary_dataset}'
   - Available: cifar10, cifar100, mnist, fashion_mnist, imdb, wikitext2, synthetic
   - For NLP: use imdb or wikitext2 (NOT vision datasets)
   - DO NOT use torchvision.datasets or create your own data loaders

2. AWS NEURON SDK REQUIREMENTS (MANDATORY - CRITICAL):
   This code MUST run on AWS Trainium using the Neuron SDK. You MUST use torch_xla (XLA) which is part of the Neuron SDK.
   
   ```python
   import torch_xla.core.xla_model as xm  # REQUIRED: Neuron SDK XLA module
   device = xm.xla_device()  # REQUIRED: Get XLA device (Trainium accelerator)
   model = model.to(device)
   inputs = inputs.to(device)  # Move ALL tensors before operations
   labels = labels.to(device)
   
   # Training loop with Neuron SDK XLA operations:
   xm.optimizer_step(optimizer)  # REQUIRED: Use XLA optimizer step (NOT optimizer.step())
   xm.mark_step()  # REQUIRED: Synchronize XLA computation after backward pass
   ```
   
   CRITICAL NEURON SDK REQUIREMENTS:
   - MUST use `torch_xla.core.xla_model as xm` - this is the Neuron SDK XLA interface
   - MUST use `xm.xla_device()` to get the Trainium device (NOT torch.device('cuda') or 'cpu')
   - MUST use `xm.optimizer_step(optimizer)` instead of `optimizer.step()` - this is Neuron SDK requirement
   - MUST call `xm.mark_step()` after each backward pass to synchronize XLA computation
   - MUST move ALL tensors to device BEFORE any operations
   - DO NOT move loss functions: `criterion = nn.CrossEntropyLoss()` (no .to(device))
   - DO NOT use CUDA: no `.cuda()`, no `device='cuda'`, no `torch.device('cuda')`
   - DO NOT use regular PyTorch device operations - this code runs on Trainium via Neuron SDK
   
   **CRITICAL: DO NOT USE NON-EXISTENT APIs** (common mistakes that cause AttributeError):
   - ❌ `xm.optimizer.SGD` - DOES NOT EXIST (use `torch.optim.SGD` then `xm.optimizer_step(optimizer)`)
   - ❌ `xm.XlaModule` - DOES NOT EXIST (use regular `nn.Module`)
   - ❌ `xm.dot()` or `xm.dot_general()` - DO NOT EXIST (use `torch.matmul()` or `torch.mm()`)
   - ❌ `xm.tensor()` - DOES NOT EXIST (use `torch.tensor()`)
   - ❌ `xm.xla_device_context()` - DOES NOT EXIST (use `device = xm.xla_device()` and `model.to(device)`)
   - ❌ `xm.optimizer_step(optimizer, sync=True)` - sync parameter DOES NOT EXIST
   
   **XLA TENSOR OPERATIONS**:
   - `tensor.size(0)` returns a tensor in XLA, use `int(tensor.size(0))` for arithmetic
   - Model outputs may be tuples - check `isinstance(model_output, tuple)` and unpack correctly
   - All standard PyTorch operations (torch.matmul, nn.Linear, etc.) work in XLA - compatibility comes from device placement

3. IMPORTS (MOST COMMON ERROR):
   ```python
   import math  # If using math.log(), math.exp(), etc.
   import torch
   import torch.nn as nn
   import torch_xla.core.xla_model as xm
   from dataset_loader import load_dataset
   ```
   - Import ALL standard library modules you use (math, random, collections, etc.)
   - This is the #1 cause of NameError failures

═══════════════════════════════════════════════════════════════════════════════
PACKAGES & ENVIRONMENT
═══════════════════════════════════════════════════════════════════════════════

AVAILABLE (Neuron SDK Environment):
- torch, torch_xla (Neuron SDK PyTorch 2.1.0 with XLA support) - REQUIRED for Trainium
- torch.nn, torch.optim (PyTorch neural network and optimizer modules)
- transformers (HuggingFace) - for tokenization - USE: 'from transformers import AutoTokenizer'
- numpy, standard library (math, random, collections, json, os, sys, etc.)
- dataset_loader (custom module)

NOTE: This code runs on AWS Trainium using the Neuron SDK. The torch_xla module is part of the Neuron SDK and provides XLA (Accelerated Linear Algebra) support for Trainium accelerators.

NOT AVAILABLE / DO NOT USE:
- transformers_xla (THIS PACKAGE DOES NOT EXIST - use 'from transformers import AutoTokenizer' instead)
- XLATokenizer (THIS CLASS DOES NOT EXIST - use AutoTokenizer instead)
- matplotlib, PIL/Pillow, pandas, scipy, sklearn, torchtext
- torchvision.datasets (use dataset_loader instead)

═══════════════════════════════════════════════════════════════════════════════
COMMON PITFALLS TO AVOID
═══════════════════════════════════════════════════════════════════════════════

1. Transformer API:
   - WRONG: `decoder(tgt, mask=mask)` → TypeError
   - CORRECT: `decoder(tgt, tgt, tgt_mask=causal_mask)` or `decoder(tgt, tgt)`

2. Tensor Operations:
   - WRONG: `torch.dot(a, b)` on batched tensors → Use `torch.sum(a * b, dim=-1)` or `torch.bmm()`
   - WRONG: `tensor[mask] = value` (in-place) → Use `torch.where()` or `torch.clamp()`

3. NLP Tokenization (CRITICAL FOR IMDB):
   - IMDB dataset returns (text_strings, labels) tuples - text is RAW STRINGS, not tokenized!
   - MUST tokenize before using:
   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   
   # In training loop:
   for batch_data in train_loader:
       if isinstance(batch_data, tuple) and len(batch_data) == 2:
           texts, labels = batch_data
           # CRITICAL: Tokenize text strings if they're strings
           if isinstance(texts, (list, tuple)) and len(texts) > 0 and isinstance(texts[0], str):
               tokenized = tokenizer(
                   list(texts),
                   padding=True,
                   truncation=True,
                   max_length=512,
                   return_tensors='pt'
               )
               inputs = tokenized['input_ids']
           else:
               inputs = texts if isinstance(texts, torch.Tensor) else torch.tensor(texts)
           
           if not isinstance(labels, torch.Tensor):
               labels = torch.tensor(labels, dtype=torch.long)
       
       inputs = inputs.to(device)
       labels = labels.to(device)
   ```
   - For WikiText-2: dataset_loader handles tokenization (vocab_size=10000)

═══════════════════════════════════════════════════════════════════════════════
CODE STRUCTURE REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

Your implementation must include:
1. Complete imports section (all modules used) - MUST include torch_xla for Neuron SDK
2. Dataset loading using dataset_loader
3. Model architecture implementing the paper's method
4. Training loop (5-10 epochs for testing) with Neuron SDK XLA device handling (xm.xla_device(), xm.optimizer_step(), xm.mark_step())
5. Evaluation/metrics (print to stdout, NO matplotlib)
6. Comprehensive comments explaining the implementation, especially Neuron SDK XLA operations

Training Loop Template (Neuron SDK XLA):
```python
# REQUIRED: Import Neuron SDK XLA module
import torch_xla.core.xla_model as xm

# REQUIRED: Get Trainium device via Neuron SDK
device = xm.xla_device()  # Returns XLA device for Trainium accelerator
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # No .to(device)!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(5):  # 5-10 epochs for testing
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_data in train_loader:
        # Handle dataset output (could be (text, label) for IMDB or (tensor, tensor))
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            inputs, labels = batch_data
            # If inputs are strings (IMDB), tokenize them first (see NLP Tokenization above)
            if isinstance(inputs, (list, tuple)) and len(inputs) > 0 and isinstance(inputs[0], str):
                tokenized = tokenizer(list(inputs), padding=True, truncation=True, 
                                  max_length=512, return_tensors='pt')
                inputs = tokenized['input_ids']
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
        else:
            inputs = batch_data
            labels = None
        
        if inputs is None or labels is None:
            continue
        
        inputs = inputs.to(device)  # Move BEFORE operations
        labels = labels.to(device)
        
        optimizer.zero_grad()
        model_output = model(inputs)
        # Handle model output (might be tuple if return_bias_scores=True)
        outputs = model_output[0] if isinstance(model_output, tuple) else model_output
        loss = criterion(outputs, labels)
        loss.backward()
        # REQUIRED: Use Neuron SDK XLA optimizer step (NOT optimizer.step())
        xm.optimizer_step(optimizer)  # Neuron SDK XLA operation
        # REQUIRED: Synchronize XLA computation (Neuron SDK requirement)
        xm.mark_step()  # Synchronize XLA computation for Trainium
        
        # CRITICAL: In XLA, loss.item() must be called AFTER xm.mark_step() to ensure synchronization
        # The loss value is now materialized and can be safely converted to Python float
        epoch_loss += float(loss.item())
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {{epoch+1}}/5, Average Loss: {{avg_loss:.4f}}")
```

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
- Complete, detailed implementation
- Brief inline comments explaining key steps (e.g., "# Apply XLA-compatible sigmoid", "# Handle tuple outputs from model")
- All necessary code to run the implementation

DO NOT include:
- Long explanations before or after the code block
- Markdown headers or sections outside code
- "Here's the code" or similar introductory text
- Dataset information text outside code
- Key features text outside code

ONLY output the code block with inline comments. The code must be immediately runnable with no manual setup required.
"""

        return base_prompt
    
    def _extract_code_from_response(self, response_text: str) -> Dict[str, str]:
        """
        Extract code and explanation from Claude's response.
        
        Args:
            response_text: Raw response from Claude
            
        Returns:
            Dictionary with extracted code and explanation
        """
        # Look for code blocks
        code_blocks = []
        explanation_parts = []
        
        lines = response_text.split('\n')
        in_code_block = False
        current_code = []
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                current_code = []
                continue
            elif line.strip().startswith('```') and in_code_block:
                in_code_block = False
                if current_code:
                    code_blocks.append('\n'.join(current_code))
                current_code = []
                continue
            elif in_code_block:
                current_code.append(line)
            else:
                explanation_parts.append(line)
        
        # Combine all code blocks
        full_code = '\n\n'.join(code_blocks) if code_blocks else ""
        
        # Clean up explanation
        explanation = '\n'.join(explanation_parts).strip()
        
        return {
            "code": full_code,
            "explanation": explanation
        }
    
    def generate_code_with_feedback(self, paper_summary: str, paper_content: Optional[str] = None, 
                                  feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate PyTorch code with iterative feedback.
        
        Args:
            paper_summary: Summary of the paper
            paper_content: Full paper content (optional)
            feedback: Previous feedback to incorporate (optional)
            
        Returns:
            Dictionary containing the generated code and metadata
        """
        try:
            # Create prompt with feedback if provided
            if feedback:
                prompt = f"""
Previous feedback: {feedback}

Please revise the PyTorch implementation based on the above feedback. Here's the paper information again:

{paper_summary}

{paper_content[:80000] if paper_content else ""}

Please provide an improved implementation addressing the feedback.
"""
            else:
                prompt = self._create_pytorch_prompt(paper_summary, paper_content)
            
            # Prepare the request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8192,  # Claude 3.5 Sonnet supports up to 8,192 tokens output
                "temperature": 0.3,  # Lower temperature for more detailed, deterministic code generation
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Bedrock request
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            # Parsing response
            response_body = json.loads(response['body'].read())
            generated_text = response_body['content'][0]['text']
            
            # Extract code from  response
            code_result = self._extract_code_from_response(generated_text)
            
            logger.info("Successfully generated revised PyTorch code")
            return {
                "success": True,
                "code": code_result["code"],
                "explanation": code_result["explanation"],
                "full_response": generated_text,
                "model_used": self.model_id,
                "iteration": True
            }
            
        except Exception as e:
            logger.error(f"Error generating code with feedback: {e}")
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "code": None,
                "explanation": None
            }
