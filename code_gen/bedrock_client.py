"""
AWS Bedrock client for interacting with Claude to generate PyTorch code.
"""

import os
import json
import logging
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
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Make the request to Bedrock
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
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
            base_prompt += f"""
Full Paper Content:
{paper_content[:8000]}  # CHANGE BASED ON MAX TOKENS ALLOWED BY BEDROCK

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
Please generate a complete PyTorch implementation that includes:

1. **Dataset Selection and Loading** (CRITICAL - READ CAREFULLY):
   You MUST use our standardized dataset loader which provides pre-cached datasets from S3.
   
   **IMPORTANT: Use the PRIMARY DATASET recommended above ({primary_dataset}).**
   
   At the START of your code, import and use the dataset loader:
   ```python
   from dataset_loader import load_dataset
   
   # Use the recommended primary dataset: {primary_dataset}
   train_loader, test_loader = load_dataset('{primary_dataset}', batch_size=128)
   ```
   
   Available datasets (ONLY THESE WORK ON TRAINIUM):
   - 'cifar10': 60K 32x32 color images, 10 classes (for CNNs, image classification) - ✅ AVAILABLE
   - 'cifar100': 60K 32x32 color images, 100 classes (harder classification) - ✅ AVAILABLE
   - 'mnist': 70K 28x28 grayscale digits (simple baselines) - ✅ AVAILABLE
   - 'fashion_mnist': 70K 28x28 grayscale fashion items - ✅ AVAILABLE
   
   ❌ UNAVAILABLE DATASETS (DO NOT USE - will cause _lzma import errors):
   - 'imdb': NOT AVAILABLE - requires _lzma module which is not available in Neuron venv
   - 'wikitext2': NOT AVAILABLE - requires _lzma module which is not available in Neuron venv
   
   IMPORTANT: For NLP tasks, use vision datasets (mnist, cifar10, etc.) or adapt the task to work with available datasets.
   
   **USE THE RECOMMENDED PRIMARY DATASET unless there's a strong reason to use an alternative.**
   
   DO NOT use torchvision.datasets or generate your own data. Use ONLY the dataset_loader.

2. **Data Preprocessing**: 
   - For VISION datasets: The dataset_loader already provides standard transforms (ToTensor, Normalize)
   - NLP datasets (imdb, wikitext2) are NOT available due to _lzma module limitations. Use vision datasets instead.
   - Example for NLP:
     ```python
     from transformers import AutoTokenizer
     tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # or appropriate model
     # In training loop, tokenize the text:
     encoded = tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt')
     ```
   - DO NOT import torchvision.transforms - it causes import errors
   - If you need custom data augmentation for vision, create transforms using torch operations

3. **Model Architecture**: Implement the main model/algorithm described in the paper

4. **Training Loop**: Include a complete training loop with proper loss functions and optimizers

5. **Evaluation**: Add evaluation metrics and testing functionality

6. **Visualization**: Print metrics and results to stdout (DO NOT use matplotlib - it's not available)
   - Use print() statements to display training progress, loss values, accuracy metrics
   - Format output clearly using f-strings with your actual variable names
   - DO NOT use plt.plot(), plt.show(), or any matplotlib functions
   - IMPORTANT: Use your actual variable names in print statements (e.g., if your loop variable is `epoch`, use `epoch` directly)

7. **Documentation**: Add comprehensive comments explaining each component

CRITICAL REQUIREMENTS FOR DATASETS:
- **ALWAYS use `from dataset_loader import load_dataset` at the top of your code**
- **USE THE RECOMMENDED PRIMARY DATASET: '{primary_dataset}'**
- **DO NOT use torchvision.datasets, HuggingFace datasets, or generate synthetic data**
- **DO NOT create download_dataset() or prepare_dataset() functions**
- The dataset_loader handles all downloading and caching automatically
- Example: `train_loader, test_loader = load_dataset('{primary_dataset}', batch_size=128)`
- Print which dataset you're using (optional): You can add a print statement to show which dataset is being used

CRITICAL - AWS TRAINIUM (NEURON SDK) REQUIREMENTS:
The code will run on AWS Trainium (trn1.2xlarge) which uses AWS Neuron SDK. You MUST ensure compatibility:

✅ REQUIRED FOR TRAINIUM (NEURON SDK):
- MUST use PyTorch/XLA (torch_xla) for training to utilize Trainium hardware acceleration
- Import: `import torch_xla.core.xla_model as xm` (comes from torch-neuronx package)
- Get device: `device = xm.xla_device()` (this gets the Trainium device)
- CRITICAL: Move ALL tensors to XLA device BEFORE any operations:
  * Models: `model = model.to(device)`
  * Input tensors: `inputs = inputs.to(device)`
  * Label tensors: `labels = labels.to(device)`
  * Any intermediate tensors created must also be on XLA device
  * Example: `tensor = torch.tensor([1, 2, 3]).to(device)` - always add `.to(device)` when creating new tensors
- CRITICAL: DO NOT move loss functions to device - they are stateless and moving them can cause crashes:
  * Correct: `criterion = torch.nn.CrossEntropyLoss()` (no .to(device))
  * Wrong: `criterion = torch.nn.CrossEntropyLoss().to(device)` (will cause segfault)
- Use `xm.optimizer_step(optimizer)` instead of `optimizer.step()` in training loops
- Use `xm.mark_step()` after each training step to synchronize with Trainium hardware
- DO NOT use CUDA-specific operations (no `.cuda()`, no `torch.cuda.is_available()`, no `device='cuda'`)
- DO NOT use `torch.device('cuda')` - Trainium does not support CUDA, only XLA devices
- ERROR TO AVOID: "Input tensor is not an XLA tensor" - this means you forgot to move a tensor to the XLA device

✅ ALLOWED PACKAGES:
- torch (PyTorch 2.1.0) - REQUIRED
- torch_xla (PyTorch/XLA for Trainium) - REQUIRED for training
- torch.nn, torch.optim, torch.utils.data - All PyTorch core modules
- transformers (HuggingFace) - Available for tokenization (use AutoTokenizer)
- datasets (HuggingFace) - Available for loading NLP datasets (used internally by dataset_loader)
- numpy - Available
- json, os, sys, time, logging - Standard library
- dataset_loader - Custom module (ALWAYS use this for datasets - it handles all dataset loading)

❌ FORBIDDEN PACKAGES (NOT AVAILABLE):
- torchvision.transforms - NOT AVAILABLE, DO NOT IMPORT (causes lzma error)
- torchvision.datasets - NOT AVAILABLE, use dataset_loader instead
- torchtext - NOT AVAILABLE, DO NOT USE
- matplotlib - NOT AVAILABLE, DO NOT USE
- PIL/Pillow - NOT AVAILABLE
- pandas - NOT AVAILABLE
- scipy - NOT AVAILABLE
- sklearn - NOT AVAILABLE
- Any other third-party packages not listed above

IMPORTANT - DATASET LOADING:
- ALWAYS use `from dataset_loader import load_dataset` - DO NOT import datasets directly
- The dataset_loader handles all dataset loading (vision from .pt files, NLP from HuggingFace Arrow format)
- For NLP tasks: The dataset_loader returns text data that you can tokenize using transformers.AutoTokenizer
- DO NOT import `datasets` from HuggingFace directly - the dataset_loader uses it internally

IMPORTANT:
- MUST use torch_xla for training to utilize Trainium hardware
- DO NOT import torchvision.transforms - it causes import errors (lzma module not available)
- DO NOT import torchtext or use torchtext.datasets
- DO NOT import matplotlib or use plt.plot/plt.show
- DO NOT import `datasets` from HuggingFace directly - use dataset_loader instead
- DO NOT use torchvision.datasets - use dataset_loader instead
- DO NOT use CUDA operations - Trainium does not support CUDA
- For visualization: Use print statements or save metrics to files (no plotting)
- For data transforms: The dataset_loader already provides transforms. If you need custom transforms, create them using torch operations (e.g., torch.nn.functional.normalize, torch.nn.functional.pad, etc.)
- For text processing: Use torch and standard Python string operations
- For NLP: Use torch.nn.Embedding and torch-based tokenization, NOT torchtext

TRAINING LOOP PATTERN FOR TRAINIUM:
```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = model.to(device)
# DO NOT move loss functions to device - they are stateless and don't need device placement
# Example: criterion = torch.nn.CrossEntropyLoss()  # No .to(device) needed

# Use 5-10 epochs for testing, not 100+ (faster execution, sufficient for testing)
num_epochs = 5  # Use small number for testing

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for inputs, labels in train_loader:
        # CRITICAL: Move ALL tensors to device BEFORE any operations
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        xm.optimizer_step(optimizer)  # Use xm.optimizer_step instead of optimizer.step()
        xm.mark_step()  # Synchronize after each step
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Print metrics INSIDE the epoch loop (epoch variable is in scope here)
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    # Print training progress - use your actual variable names
    # Example: print(f"Epoch completed, average loss: {{avg_loss:.4f}}")
```

Requirements:
- Use modern PyTorch practices (PyTorch 2.1.0 features)
- Include proper error handling and logging
- Make the code modular and reusable
- Add type hints where appropriate
- Handle edge cases and provide fallbacks
- THE CODE MUST BE IMMEDIATELY RUNNABLE - no manual dataset setup required
- Use ONLY the packages listed in the ALLOWED list above
- CRITICAL: Ensure ALL tensors are moved to XLA device before any operations to avoid "Input tensor is not an XLA tensor" errors

Please structure your response as follows:

## PyTorch Implementation

### Overview
[Brief explanation of what the code implements]

### Dependencies
[List of required packages including dataset download libraries]

### Code Implementation
```python
[Complete PyTorch code here with dataset downloading/generation]
```

### Usage Example
[Example of how to use the code - should just work when run]

### Dataset Information
[Specify which dataset from dataset_loader you selected and WHY it matches this paper's requirements]

### Key Features
[List of main features implemented]

Please ensure the code is production-ready, follows best practices, and can be run immediately after installing dependencies.
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

{paper_content[:4000] if paper_content else ""}

Please provide an improved implementation addressing the feedback.
"""
            else:
                prompt = self._create_pytorch_prompt(paper_summary, paper_content)
            
            # Prepare the request body
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
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
