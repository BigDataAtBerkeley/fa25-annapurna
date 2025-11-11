#!/usr/bin/env python3
"""
Code Review Agent for Generated PyTorch Code

This agent analyzes generated code and iteratively fixes common issues
before the code is tested on Trn
"""

import logging
import re
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from bedrock_client import BedrockClient
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CodeReviewAgent:
    """
    Agent that reviews and fixes generated PyTorch code iteratively.
    """
    
    def __init__(self, bedrock_client: Optional[BedrockClient] = None):
        """
        Initialize the code review agent.
        
        Args:
            bedrock_client: BedrockClient instance (creates new one if None)
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.max_iterations = 1  # for now we only iterate once, will up this
        self.fix_history = []  # Track fixes applied
    
    def review_and_fix_code(self, code: str, dataset_name: str = None) -> Dict[str, Any]:
        """
        Review code and iteratively fix issues.
        
        Args:
            code: Generated PyTorch code to review
            dataset_name: Name of dataset being used (e.g., 'imdb', 'cifar10')
            
        Returns:
            Dictionary with:
            - 'code': Fixed code
            - 'fixes_applied': List of fixes made
            - 'iterations': Number of fix iterations
            - 'success': Whether code was successfully fixed
        """
        logger.info("Starting code review and fix process...")
        
        current_code = code
        fixes_applied = []
        iteration = 0
        
        # static checks first before AI
        static_issues = self._check_static_issues(current_code, dataset_name)
        if static_issues:
            logger.info(f"Found {len(static_issues)} static issues, applying quick fixes...")
            current_code, quick_fixes = self._apply_quick_fixes(current_code, static_issues, dataset_name)
            fixes_applied.extend(quick_fixes)
        
        # AI analysis 
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Code review iteration {iteration}/{self.max_iterations}")
            
            # Analyze code for issues
            issues = self._analyze_code_with_ai(current_code, dataset_name, fixes_applied)
            
            if not issues or issues.get('no_issues', False):
                logger.info("✅ No issues found - code is ready!")
                break
            
            # Record issues found (even if fix fails, we want to track what was found)
            issues_found = issues.get('issues', [])
            fixes_needed = issues.get('fixes_needed', [])
            
            if issues_found:
                logger.info(f"Found {len(issues_found)} issues: {', '.join(issues_found[:3])}...")
            
            # Fix issues
            fixed_code = self._fix_code_with_ai(current_code, issues, dataset_name)
            
            if fixed_code and fixed_code != current_code:
                fixes_applied.append({
                    'iteration': iteration,
                    'issues_found': issues_found,
                    'fixes': fixes_needed  # Use fixes_needed, not fixes
                })
                current_code = fixed_code
                logger.info(f"✅ Applied fixes in iteration {iteration} - code updated")
            elif fixed_code is None:
                # Fix failed - record issues but warn
                logger.error(f"❌ Code fix failed - AI could not generate fixed code")
                logger.error(f"   Issues found: {len(issues_found)}")
                # Still record the issues so we know what was wrong
                fixes_applied.append({
                    'iteration': iteration,
                    'issues_found': issues_found,
                    'fixes': fixes_needed,
                    'fix_failed': True  # Mark that fix attempt failed
                })
                # Don't break - try to continue with original code, but log the failure
                logger.warning("⚠️ Continuing with original code (fix failed)")
                break
            else:
                # Fix returned same code - might mean no changes needed or fix failed
                logger.warning(f"⚠️ Fix returned same code - no changes applied")
                # Still record issues found
                fixes_applied.append({
                    'iteration': iteration,
                    'issues_found': issues_found,
                    'fixes': fixes_needed,
                    'no_changes': True  # Mark that no changes were made
                })
                break
        
        return {
            'code': current_code,
            'fixes_applied': fixes_applied,
            'iterations': iteration,
            'success': len(fixes_applied) > 0 or iteration == 1
        }
    
    def _check_static_issues(self, code: str, dataset_name: str = None) -> List[str]:
        """
        Quick static analysis for common issues.
        
        Returns:
            List of issue descriptions
        """
        issues = []
        
        # Check if code appears incomplete/truncated
        if code:
            # Check for incomplete code patterns
            code_stripped = code.rstrip()
            
            # Check for unclosed brackets/parens/braces
            open_parens = code.count('(') - code.count(')')
            open_brackets = code.count('[') - code.count(']')
            open_braces = code.count('{') - code.count('}')
            
            if open_parens > 0:
                issues.append(f"Code appears incomplete: {open_parens} unclosed parenthesis")
            if open_brackets > 0:
                issues.append(f"Code appears incomplete: {open_brackets} unclosed brackets")
            if open_braces > 0:
                issues.append(f"Code appears incomplete: {open_braces} unclosed braces")
            
            # Check if code ends with incomplete statement
            incomplete_endings = [',', '(', '[', '{', '\\', '=']
            if code_stripped and code_stripped[-1] in incomplete_endings:
                issues.append(f"Code appears incomplete: ends with '{code_stripped[-1]}' (likely truncated)")
            
            # Check for incomplete function/class definitions (no body)
            lines = code.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Check for def/class without body
                if (stripped.startswith('def ') or stripped.startswith('class ')) and stripped.endswith(':'):
                    # Check if next non-empty line is at same or less indentation (no body)
                    if i + 1 < len(lines):
                        next_lines = [l for l in lines[i+1:] if l.strip()]
                        if next_lines:
                            next_line = next_lines[0]
                            if not next_line.startswith(' ') or len(next_line) - len(next_line.lstrip()) <= len(line) - len(line.lstrip()):
                                issues.append(f"Code appears incomplete: {stripped} has no body")
                                break
            
            # Check if code is suspiciously short for a complete implementation
            # (This is a heuristic - very short code might be incomplete)
            if len(code.split('\n')) < 20 and ('for ' in code or 'def ' in code or 'class ' in code):
                issues.append("Code appears suspiciously short - may be incomplete or truncated")
            
            # Check if code ends with incomplete comment (clear sign of truncation)
            if code_stripped.endswith('#') or code_stripped.endswith('# '):
                # Check if it's a comment that looks incomplete
                last_line = lines[-1].strip() if lines else ''
                if last_line.startswith('#') and len(last_line) < 10:
                    issues.append("Code appears incomplete: ends with incomplete comment (likely truncated by max_tokens)")
            
            # Check if last line is incomplete (doesn't end with newline or proper statement)
            if lines and lines[-1].strip() and not lines[-1].rstrip().endswith((':', ';', ')', ']', '}')):
                # Check if it looks like it was cut off mid-statement
                last_line = lines[-1].strip()
                if any(last_line.endswith(char) for char in [',', '(', '[', '{', '=', '+', '-', '*', '/', '%']):
                    issues.append(f"Code appears incomplete: last line ends with '{last_line[-1]}' (likely truncated)")
        
        # Check for IMDB dataset without tokenization
        if dataset_name == 'imdb' or 'imdb' in code.lower():
            if 'load_dataset' in code and 'imdb' in code.lower():
                # Check if tokenization is present
                has_tokenizer = 'AutoTokenizer' in code or 'tokenizer' in code.lower()
                has_tokenization_in_loop = 'tokenizer(' in code and 'train_loader' in code
                
                if has_tokenizer and not has_tokenization_in_loop:
                    issues.append("IMDB dataset requires tokenization in training loop but tokenization code is missing")
                elif not has_tokenizer:
                    issues.append("IMDB dataset requires AutoTokenizer import and tokenization")
        
        # Check for model output tuple handling
        if 'return_bias_scores' in code or 'return_attention' in code:
            if 'isinstance(model_output, tuple)' not in code and 'isinstance(outputs, tuple)' not in code:
                issues.append("Model may return tuple but code doesn't handle tuple outputs")
        
        # Check for XLA device usage
        if 'xm.xla_device()' in code:
            if 'xm.optimizer_step' not in code:
                issues.append("XLA device used but xm.optimizer_step() missing in training loop")
            if 'xm.mark_step' not in code:
                issues.append("XLA device used but xm.mark_step() missing in training loop")
            
            # Check for xm.rendezvous() - must have a tag argument on Trainium/Neuron
            if 'xm.rendezvous()' in code:
                # Check if it's called without arguments (will cause IndexError on Trainium)
                rendezvous_pattern = r'xm\.rendezvous\s*\(\s*\)'
                if re.search(rendezvous_pattern, code):
                    issues.append("CRITICAL: xm.rendezvous() called without required tag argument - will cause IndexError on Trainium. Must be xm.rendezvous('tag_name')")
            
            # Check for XLA tensor size issues (common bug: using tensor.size(0) directly as int)
            if '.size(0)' in code or '.shape[0]' in code:
                # Check if there are patterns like "count += inputs.size(0)" or "batch_size = inputs.size(0)"
                # that don't convert to int
                size_patterns = [
                    r'count\s*\+=\s*\w+\.size\(0\)',
                    r'batch_size\s*=\s*\w+\.size\(0\)',
                    r'count\s*\+=\s*\w+\.shape\[0\]',
                    r'batch_size\s*=\s*\w+\.shape\[0\]',
                ]
                for pattern in size_patterns:
                    matches = re.findall(pattern, code)
                    for match in matches:
                        # Check if int() conversion is nearby (within 5 lines)
                        match_line = code.find(match)
                        context = code[max(0, match_line-200):match_line+200]
                        if 'int(' not in context and 'int(' not in match:
                            issues.append(f"XLA tensor size used without int() conversion: {match.strip()} - in XLA, tensor.size(0) returns a tensor, not Python int")
                            break
        
        # Check for missing imports
        if 'math.' in code and 'import math' not in code:
            issues.append("Missing import: math")
        if 'random.' in code and 'import random' not in code:
            issues.append("Missing import: random")
        if 'collections.' in code and 'from collections' not in code:
            issues.append("Missing import: collections")
        
        # Check for invalid/non-existent imports
        if 'transformers_xla' in code or 'from transformers_xla' in code or 'import transformers_xla' in code:
            issues.append("CRITICAL: transformers_xla package does not exist. Use 'from transformers import AutoTokenizer' instead. transformers_xla is not a real package.")
        if 'XLATokenizer' in code:
            issues.append("CRITICAL: XLATokenizer does not exist. Use AutoTokenizer from transformers package instead.")
        
        return issues
    
    def _apply_quick_fixes(self, code: str, issues: List[str], dataset_name: str = None) -> Tuple[str, List[str]]:
        """
        Apply quick fixes that don't require AI.
        
        Returns:
            Tuple of (fixed_code, list_of_fixes_applied)
        """
        fixed_code = code
        fixes_applied = []
        
        # Fix invalid transformers_xla imports
        if 'transformers_xla' in fixed_code or 'XLATokenizer' in fixed_code:
            # Replace transformers_xla imports with correct transformers imports
            fixed_code = fixed_code.replace('from transformers_xla import XLATokenizer', 'from transformers import AutoTokenizer')
            fixed_code = fixed_code.replace('import transformers_xla', '# transformers_xla does not exist, using transformers instead')
            fixed_code = fixed_code.replace('XLATokenizer', 'AutoTokenizer')
            if 'transformers_xla' in fixed_code or 'XLATokenizer' in fixed_code:
                fixes_applied.append("Fixed invalid transformers_xla/XLATokenizer imports - replaced with AutoTokenizer from transformers")
        
        # Add missing imports
        if any('Missing import' in issue for issue in issues):
            imports_to_add = []
            if 'Missing import: math' in str(issues):
                imports_to_add.append('import math')
            if 'Missing import: random' in str(issues):
                imports_to_add.append('import random')
            if 'Missing import: collections' in str(issues):
                imports_to_add.append('from collections import defaultdict')
            
            if imports_to_add:
                # Find first import line and add after it
                import_pattern = r'(^import torch\n|^from torch import)'
                match = re.search(import_pattern, fixed_code, re.MULTILINE)
                if match:
                    insert_pos = match.end()
                    new_imports = '\n'.join(imports_to_add) + '\n'
                    fixed_code = fixed_code[:insert_pos] + new_imports + fixed_code[insert_pos:]
                    fixes_applied.append(f"Added missing imports: {', '.join(imports_to_add)}")
        
        # Fix xm.rendezvous() without arguments (CRITICAL - causes IndexError on Trainium)
        if 'xm.rendezvous()' in fixed_code:
            # Replace xm.rendezvous() with xm.rendezvous('init')
            rendezvous_pattern = r'xm\.rendezvous\s*\(\s*\)'
            if re.search(rendezvous_pattern, fixed_code):
                fixed_code = re.sub(rendezvous_pattern, "xm.rendezvous('init')", fixed_code)
                fixes_applied.append("Fixed xm.rendezvous() - added required tag argument 'init'")
                logger.info("✅ Quick fix: Added tag argument to xm.rendezvous()")
        
        return fixed_code, fixes_applied
    
    def _analyze_code_with_ai(self, code: str, dataset_name: str = None, 
                              previous_fixes: List[Dict] = None) -> Dict[str, Any]:
        """
        Use AI to analyze code for issues.
        
        Returns:
            Dictionary with 'issues' list and 'fixes' list
        """
        fix_history = ""
        if previous_fixes:
            fix_history = "\nPrevious fixes applied:\n"
            for fix in previous_fixes:
                fix_history += f"- Iteration {fix['iteration']}: {', '.join(fix.get('fixes', []))}\n"
        
        analysis_prompt = f"""You are an expert code reviewer for PyTorch code that will run on AWS Trainium using the Neuron SDK (XLA devices).

Your task: Review this PyTorch code and identify ANY issues that would prevent it from running correctly on Trainium with the Neuron SDK. Look for ALL types of errors, not just common ones.

CRITICAL: This code MUST use the Neuron SDK (torch_xla) for Trainium execution. Ensure all XLA operations are correct.

```python
{code}
```

Dataset being used: {dataset_name or 'unknown'}

{fix_history}

COMPREHENSIVE REVIEW CHECKLIST - Check for ALL of these:

1. **Neuron SDK / XLA / Trainium Compatibility (CRITICAL):**
   - MUST use torch_xla.core.xla_model as xm (Neuron SDK XLA module)
   - MUST use xm.xla_device() to get Trainium device (NOT torch.device('cuda') or 'cpu')
   - MUST use xm.optimizer_step(optimizer) instead of optimizer.step() (Neuron SDK requirement)
   - MUST call xm.mark_step() after each backward pass (Neuron SDK synchronization)
   - All tensor operations compatible with XLA (no in-place operations on indexed tensors)
   - Tensor size operations: tensor.size(0) returns a tensor in XLA, must use int() for arithmetic
   - No CUDA-specific code (.cuda(), device='cuda', torch.device('cuda'))
   - Ensure all tensors moved to XLA device before operations

2. **Data Handling:**
   - IMDB dataset: Returns (text_strings, labels) - text must be tokenized before use
   - DataLoader iteration: Handle both (tensor, tensor) and (text_strings, labels) formats
   - All tensors moved to device before operations
   - Proper handling of batch data unpacking

3. **Model Output Handling:**
   - Model may return tuples - check isinstance() before using
   - Proper unpacking of model outputs
   - Handle optional return values (e.g., return_bias_scores=True)

4. **Type Errors:**
   - Mixing tensors with Python ints/floats in arithmetic
   - Calling methods on wrong types (e.g., .to() on list, .item() on non-scalar)
   - Indexing issues with XLA tensors

5. **Import Errors:**
   - All used modules imported (math, random, collections, etc.)
   - Correct import paths

6. **Logic Errors:**
   - Division by zero or None
   - Uninitialized variables
   - Incorrect tensor shapes/dimensions
   - Wrong device placement

7. **Runtime Errors:**
   - AttributeError (calling methods on wrong types)
   - TypeError (wrong argument types)
   - ValueError (wrong tensor shapes, dimensions)
   - IndexError (out of bounds access)

8. **XLA-Specific Gotchas:**
   - Using tensor.size(0) directly in arithmetic without int() conversion
   - Using tensor values in Python control flow without .item()
   - In-place operations that XLA doesn't support
   - Loss functions not moved to device (they shouldn't be)

IMPORTANT: Be thorough. Look for ANY error that would cause the code to fail, not just the common ones listed above. Think like a compiler/linter - catch everything.

Respond in JSON format:
{{
    "no_issues": false,
    "issues": ["detailed issue description 1", "detailed issue description 2", ...],
    "fixes_needed": ["specific fix instruction 1", "specific fix instruction 2", ...],
    "critical": true/false
}}

If no issues found, set "no_issues": true."""

        try:
            # Retry logic for Bedrock throttling
            max_retries = 5
            base_delay = 2
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.bedrock_client.client.invoke_model(
                        modelId=self.bedrock_client.model_id,
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 8192,  # Claude 3.5 Sonnet supports up to 8,192 tokens output
                            "temperature": 0.3,  # Lower temperature for more thorough analysis
                            "messages": [
                                {
                                    "role": "user",
                                    "content": analysis_prompt
                                }
                            ]
                        }),
                        contentType="application/json"
                    )
                    break  # Success
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    # Retry on throttling or service unavailability (both are often transient)
                    retryable_errors = ['ThrottlingException', 'ServiceUnavailableException']
                    if error_code in retryable_errors and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Code review analysis {error_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise
            
            if response is None:
                raise Exception("Failed to get response after retries")
            
            response_body = json.loads(response['body'].read())
            analysis_text = response_body['content'][0]['text']
            
            # Try to extract JSON from response
            # Look for JSON block (may be in code block or plain text)
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON in code block
                r'```\s*(\{.*?"no_issues".*?\})\s*```',  # JSON in generic code block
                r'(\{[^{}]*"no_issues"[^{}]*\})',  # JSON in text
            ]
            
            analysis_result = None
            for pattern in json_patterns:
                json_match = re.search(pattern, analysis_text, re.DOTALL)
                if json_match:
                    try:
                        analysis_result = json.loads(json_match.group(1))
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not analysis_result:
                # Fallback: parse manually from text
                text_lower = analysis_text.lower()
                has_issues = any(word in text_lower for word in ['issue', 'problem', 'error', 'bug', 'fix'])
                no_issues = any(phrase in text_lower for phrase in ['no issues', 'no problems', 'looks good', 'correct'])
                
                analysis_result = {
                    "no_issues": no_issues and not has_issues,
                    "issues": [],
                    "fixes_needed": []
                }
                
                # Try to extract issue descriptions
                if has_issues and not no_issues:
                    # Look for bullet points or numbered lists
                    issue_lines = re.findall(r'[-*•]\s*(.+?)(?:\n|$)', analysis_text, re.MULTILINE)
                    if issue_lines:
                        analysis_result["issues"] = issue_lines[:5]  # Limit to 5 issues
                    else:
                        analysis_result["issues"] = ["Issues detected - see full analysis"]
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in AI code analysis: {e}")
            return {"no_issues": True, "issues": [], "fixes_needed": []}
    
    def _fix_code_with_ai(self, code: str, issues: Dict[str, Any], 
                          dataset_name: str = None) -> Optional[str]:
        """
        Use AI to fix code issues.
        
        Returns:
            Fixed code or None if fix failed
        """
        issues_list = issues.get('issues', [])
        fixes_needed = issues.get('fixes_needed', [])
        
        if not issues_list and not fixes_needed:
            return code
        
        fix_prompt = f"""Fix the following PyTorch code to work on AWS Trainium using the Neuron SDK (XLA devices).

CRITICAL: This code MUST use the Neuron SDK (torch_xla) for Trainium execution. All XLA operations must be correct.

Issues identified:
{chr(10).join(f'- {issue}' for issue in issues_list)}
{chr(10).join(f'- {fix}' for fix in fixes_needed)}

Original code:
```python
{code}
```

Dataset: {dataset_name or 'unknown'}

CRITICAL FIXES NEEDED:
1. If using IMDB dataset: Add tokenization in training/test loops (IMDB returns text strings, not tokenized tensors)
2. Handle model tuple outputs: Check if model returns tuple and extract first element
3. XLA tensor size conversion (CRITICAL - MOST COMMON BUG):
   - Find ALL uses of tensor.size(0) or tensor.shape[0] in arithmetic operations
   - WRONG: count += inputs.size(0) or batch_size = inputs.size(0) or scores / count where count came from size(0)
   - CORRECT: count += int(inputs.size(0)) or batch_size = int(inputs.size(0))
   - Before division: count = int(count) if hasattr(count, 'item') else int(count) before scores / count
   - In XLA, tensor.size(0) returns a tensor, NOT a Python int - this causes AttributeError
4. Ensure all tensors are moved to XLA device (xm.xla_device()) before operations
5. MUST use xm.optimizer_step(optimizer) instead of optimizer.step() (Neuron SDK requirement)
6. MUST call xm.mark_step() after each backward pass (Neuron SDK synchronization)
7. xm.rendezvous() MUST have a tag argument on Trainium/Neuron:
   - WRONG: xm.rendezvous()  (will cause IndexError: tuple index out of range)
   - CORRECT: xm.rendezvous('init') or xm.rendezvous('training') or any string tag
8. Add any missing imports

IMPORTANT: Return the COMPLETE fixed Python code. Include ALL imports, ALL functions, and ALL the main execution code. Do not return partial code or explanations - return the full working code that can be executed directly.

Return ONLY the complete fixed Python code in a code block. Do not include explanations outside the code block."""

        try:
            # Retry logic for Bedrock throttling
            max_retries = 5
            base_delay = 2
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.bedrock_client.client.invoke_model(
                        modelId=self.bedrock_client.model_id,
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 8192,  # Claude 3.5 Sonnet supports up to 8,192 tokens output
                            "temperature": 0.3,  # Lower temperature for more detailed fixes
                            "messages": [
                                {
                                    "role": "user",
                                    "content": fix_prompt
                                }
                            ]
                        }),
                        contentType="application/json"
                    )
                    break  # Success
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    # Retry on throttling or service unavailability (both are often transient)
                    retryable_errors = ['ThrottlingException', 'ServiceUnavailableException']
                    if error_code in retryable_errors and attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Code review fix {error_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise
            
            if response is None:
                raise Exception("Failed to get response after retries")
            
            response_body = json.loads(response['body'].read())
            fixed_text = response_body['content'][0]['text']
            
            logger.info(f"AI fix response length: {len(fixed_text)} characters")
            
            # Extract code from response - try multiple patterns
            code_patterns = [
                r'```python\n(.*?)\n```',  # Python code block with python tag
                r'```python\n(.*?)```',  # Python code block without closing newline
                r'```\n(.*?)\n```',  # Generic code block
                r'```(.*?)```',  # Code block without newlines
            ]
            
            extracted_code = None
            for pattern in code_patterns:
                code_match = re.search(pattern, fixed_text, re.DOTALL)
                if code_match:
                    candidate = code_match.group(1).strip()
                    # Validate it's actually code (has imports or def/class)
                    if 'import' in candidate or 'def ' in candidate or 'class ' in candidate:
                        extracted_code = candidate
                        logger.info(f"✅ Extracted code using pattern: {pattern[:20]}...")
                        break
            
            # Fallback: if no code block found, check if entire response looks like code
            if not extracted_code:
                # Remove markdown formatting if present
                cleaned_text = fixed_text.strip()
                # Remove leading/trailing markdown code block markers if they exist
                if cleaned_text.startswith('```'):
                    lines = cleaned_text.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]
                    cleaned_text = '\n'.join(lines).strip()
                
                # Check if it looks like code
                if ('import' in cleaned_text or 'def ' in cleaned_text or 'class ' in cleaned_text) and len(cleaned_text) > 100:
                    extracted_code = cleaned_text
                    logger.info("✅ Using entire response as code (no code block markers found)")
            
            if extracted_code:
                # Validate the extracted code is different and substantial
                if extracted_code != code:
                    if len(extracted_code) > 50:  # Must be substantial
                        logger.info(f"✅ Successfully extracted fixed code ({len(extracted_code)} chars)")
                        return extracted_code
                    else:
                        logger.warning(f"Extracted code too short ({len(extracted_code)} chars) - likely incomplete")
                else:
                    logger.warning("Extracted code is identical to original - no changes made")
            else:
                logger.error("Could not extract fixed code from AI response")
                logger.error(f"Response preview: {fixed_text[:500]}...")
            
            # If we can't extract code, return None (fix failed)
            return None
            
        except Exception as e:
            logger.error(f"Error in AI code fix: {e}")
            return None

