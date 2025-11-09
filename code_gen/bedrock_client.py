"""
AWS Bedrock client for interacting with Claude to generate PyTorch code.
"""

import os
import json
import logging
import time
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
        
        # Initialize Bedrock client
        self.client = boto3.client("bedrock-runtime", region_name=self.aws_region)
        
        logger.info(f"Bedrock client initialized with model: {self.model_id}")
    
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
                "max_tokens": 32000,  # Claude Sonnet 4.5 supports up to 64,000 tokens
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Make the request to Bedrock with retry logic for throttling
            max_retries = 5
            base_delay = 2  # Start with 2 seconds
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
                    if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Bedrock throttled, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise  # Re-raise if not throttling or out of retries
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            generated_text = response_body['content'][0]['text']
            
            # Extract code from the response
            code_result = self._extract_code_from_response(generated_text)
            
            logger.info("Successfully generated PyTorch code")
            return {
                "success": True,
                "code": code_result["code"],
                "explanation": code_result["explanation"],
                "full_response": generated_text,
                "model_used": self.model_id
            }
            
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
            # Claude Sonnet 4.5 has 200k token context window, so we can include much more
            base_prompt += f"""
Full Paper Content:
{paper_content[:80000]}

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
🚨 CRITICAL REQUIREMENTS - READ CAREFULLY
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

2. TRAINIUM/XLA REQUIREMENTS (MANDATORY):
   ```python
   import torch_xla.core.xla_model as xm
   device = xm.xla_device()
   model = model.to(device)
   inputs = inputs.to(device)  # Move ALL tensors before operations
   labels = labels.to(device)
   
   # Training loop:
   xm.optimizer_step(optimizer)  # NOT optimizer.step()
   xm.mark_step()  # After each backward pass
   ```
   - CRITICAL: Move ALL tensors to device BEFORE any operations
   - DO NOT move loss functions: `criterion = nn.CrossEntropyLoss()` (no .to(device))
   - DO NOT use CUDA: no `.cuda()`, no `device='cuda'`

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
📦 PACKAGES & ENVIRONMENT
═══════════════════════════════════════════════════════════════════════════════

✅ AVAILABLE:
- torch, torch_xla, torch.nn, torch.optim (PyTorch 2.1.0)
- transformers (HuggingFace) - for tokenization
- numpy, standard library (math, random, collections, json, os, sys, etc.)
- dataset_loader (custom module)

❌ NOT AVAILABLE:
- matplotlib, PIL/Pillow, pandas, scipy, sklearn, torchtext
- torchvision.datasets (use dataset_loader instead)

═══════════════════════════════════════════════════════════════════════════════
⚠️ COMMON PITFALLS TO AVOID
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
📝 CODE STRUCTURE REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════════

Your implementation must include:
1. Complete imports section (all modules used)
2. Dataset loading using dataset_loader
3. Model architecture implementing the paper's method
4. Training loop (5-10 epochs for testing) with XLA device handling
5. Evaluation/metrics (print to stdout, NO matplotlib)
6. Comprehensive comments explaining the implementation

Training Loop Template:
```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
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
        xm.optimizer_step(optimizer)
        xm.mark_step()  # Synchronize XLA computation
        
        # CRITICAL: In XLA, loss.item() must be called AFTER xm.mark_step() to ensure synchronization
        # The loss value is now materialized and can be safely converted to Python float
        epoch_loss += float(loss.item())
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {{epoch+1}}/5, Average Loss: {{avg_loss:.4f}}")
```

═══════════════════════════════════════════════════════════════════════════════
📤 OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

Structure your response as:

## PyTorch Implementation

### Overview
[What the code implements]

### Code Implementation
```python
[Complete, runnable PyTorch code]
```

### Dataset Information
[Which dataset you used and why it matches the paper]

### Key Features
[Main features implemented]

The code must be immediately runnable with no manual setup required.
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
                "max_tokens": 32000,  # Claude Sonnet 4.5 supports up to 64,000 tokens
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # BEdrock requestt
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
