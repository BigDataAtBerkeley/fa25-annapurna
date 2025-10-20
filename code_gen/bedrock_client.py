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
    
    def generate_pytorch_code(self, paper_summary: str, paper_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate PyTorch code based on a research paper.
        
        Args:
            paper_summary: Summary of the paper (title, authors, abstract)
            paper_content: Full paper content (optional)
            
        Returns:
            Dictionary containing the generated code and metadata
        """
        try:
            # Prepare the prompt for Claude
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
    
    def _create_pytorch_prompt(self, paper_summary: str, paper_content: Optional[str] = None) -> str:
        """
        ACTUAL PROMPT SENT TO BEDROCK TO GENERATE CODE
        
        Args:
            paper_summary: Summary of the paper
            paper_content: Full paper content (optional)
            
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

        base_prompt += """
Please generate a complete PyTorch implementation that includes:

1. **Data Loading and Preprocessing**: Create appropriate data loaders and preprocessing pipelines
2. **Model Architecture**: Implement the main model/algorithm described in the paper
3. **Training Loop**: Include a complete training loop with proper loss functions and optimizers
4. **Evaluation**: Add evaluation metrics and testing functionality
5. **Visualization**: Include plotting and visualization code where appropriate
6. **Documentation**: Add comprehensive comments explaining each component

Requirements:
- Use modern PyTorch practices (PyTorch 2.0+ features when applicable)
- Include proper error handling and logging
- Make the code modular and reusable
- Add type hints where appropriate
- Include example usage and configuration
- Handle edge cases and provide fallbacks
- Use appropriate libraries (torch, torchvision, numpy, matplotlib, etc.)

Please structure your response as follows:

## PyTorch Implementation

### Overview
[Brief explanation of what the code implements]

### Dependencies
[List of required packages]

### Code Implementation
```python
[Complete PyTorch code here]
```

### Usage Example
[Example of how to use the code]

### Key Features
[List of main features implemented]

Please ensure the code is production-ready and follows best practices.
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
