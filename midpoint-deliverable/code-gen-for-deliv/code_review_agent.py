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
import os
import requests
from typing import Dict, Any, Optional, List, Tuple
from bedrock_client import BedrockClient
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CodeReviewAgent:
    """
    Agent that reviews and fixes generated PyTorch code iteratively.
    """
    
    def __init__(self, bedrock_client: Optional[BedrockClient] = None, 
                 enable_execution_testing: bool = False,
                 trainium_endpoint: Optional[str] = None,
                 execution_timeout: int = 120,
                 opensearch_client = None,
                 dynamo_client = None):
        """
        Initialize the code review agent.
        
        Args:
            bedrock_client: BedrockClient instance (creates new one if None)
            enable_execution_testing: If True, execute code on Trainium during review to get real errors
            trainium_endpoint: Trainium executor endpoint (e.g., "http://1.2.3.4:8000")
            execution_timeout: Timeout for execution tests in seconds (default 2 minutes)
                              This is SHORT because we only need to catch immediate errors.
                              If code takes longer, that's a problem we should know about.
            opensearch_client: OpenSearchClient instance for similar paper search (optional)
            dynamo_client: DynamoClient instance for error database (optional)
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.max_iterations = int(os.getenv('CODE_REVIEW_MAX_ITERATIONS', '6'))  # Max iterations (default 6)
        self.fix_history = []  # Track fixes applied
        self.enable_execution_testing = enable_execution_testing
        self.trainium_endpoint = trainium_endpoint or os.getenv('TRAINIUM_ENDPOINT')
        self.execution_timeout = execution_timeout
        self.stability_time = int(os.getenv('CODE_REVIEW_STABILITY_TIME', '120'))  # 2 minutes - if code runs this long without errors, consider it stable
        
        # Store clients for similar paper error retrieval
        self.opensearch_client = opensearch_client
        self.dynamo_client = dynamo_client
    
    def _get_trainium_error_reference(self) -> str:
        """
        Get a concise reference of real Trainium execution errors.
        This is shared between analysis and fix prompts to avoid duplication.
        """
        return """**⚠️ CRITICAL: REAL TRAINIUM ERRORS - MUST PREVENT/FIX:**

1. `AttributeError: 'ellipsis' object has no attribute 'X'` - Variable assigned to `...` (e.g., `base_model = ...`)
2. `AttributeError: module 'torch_xla.core.xla_model' has no attribute 'XlaModule'` - Use `nn.Module`, not `xm.XlaModule`
3. `AttributeError: module 'torch_xla.core.xla_model' has no attribute 'dot_general'` - Use `torch.matmul()`, not `xm.dot_general()`
4. `AttributeError: module 'torch_xla.core.xla_model' has no attribute 'tensor'` - Use `torch.tensor()`, not `xm.tensor()`
5. `AttributeError: module 'torch_xla.core.xla_model' has no attribute 'scalar_tensor_to_python_scalar'` - Use `.item()` or `int()`
6. `TypeError: optimizer_step() got an unexpected keyword argument 'sync'` - Use `xm.optimizer_step(optimizer)` (no sync param)
7. `AttributeError: module 'torch_xla.core.xla_model' has no attribute 'xla_device_context'` - Remove context manager, use direct calls

**Quick fixes:**
- ❌ `class LoRA(xm.XlaModule):` → ✅ `class LoRA(nn.Module):`
- ❌ `xm.dot_general(x, w)` → ✅ `torch.matmul(x, w)`
- ❌ `xm.tensor(0, ...)` → ✅ `torch.tensor(0, ...).to(device)`
- ❌ `xm.optimizer_step(opt, sync=True)` → ✅ `xm.optimizer_step(opt)`
- ❌ `base_model = ...` → ✅ `base_model = nn.Sequential(...)`"""
    
    def review_and_fix_code(self, code: str, dataset_name: str = None, 
                           paper_id: str = None, paper_title: str = None) -> Dict[str, Any]:
        """
        Review code and iteratively fix issues.
        
        Args:
            code: Generated PyTorch code to review
            dataset_name: Name of dataset being used (e.g., 'imdb', 'cifar10')
            paper_id: Paper ID for execution testing (optional, required if enable_execution_testing=True)
            paper_title: Paper title for execution testing (optional)
            
        Returns:
            Dictionary with:
            - 'code': Fixed code
            - 'fixes_applied': List of fixes made
            - 'iterations': Number of fix iterations
            - 'success': Whether code was successfully fixed
            - 'execution_tests': List of execution test results (if enabled)
        """
        logger.info("Starting code review and fix process...")
        review_start_time = time.time()
        
        current_code = code
        fixes_applied = []
        iteration = 0
        
        # Get similar paper errors in first iteration (if paper_id and clients are available)
        similar_paper_errors = []
        if iteration == 0 and paper_id and self.opensearch_client and self.dynamo_client:
            try:
                logger.info(f"🔍 Finding similar papers to {paper_id} for proactive error checking...")
                # Get paper to extract abstract
                paper = self.opensearch_client.get_paper_by_id(paper_id)
                if paper:
                    abstract = paper.get('abstract', '')
                    if abstract:
                        # Search for top 3 similar papers
                        similar_papers = self.opensearch_client.search_similar_papers_by_abstract(
                            abstract=abstract,
                            exclude_id=paper_id,
                            size=3
                        )
                        
                        if similar_papers:
                            similar_paper_ids = [p.get('_id') for p in similar_papers if p.get('_id')]
                            logger.info(f"Found {len(similar_paper_ids)} similar papers: {', '.join(similar_paper_ids)}")
                            
                            # Get errors from similar papers
                            similar_paper_errors = self.dynamo_client.get_errors_for_paper_ids(similar_paper_ids)
                            if similar_paper_errors:
                                logger.info(f"✅ Retrieved {len(similar_paper_errors)} errors from similar papers for proactive checking")
                            else:
                                logger.info("No errors found in similar papers")
                        else:
                            logger.info("No similar papers found")
                    else:
                        logger.warning(f"Paper {paper_id} has no abstract - cannot search for similar papers")
                else:
                    logger.warning(f"Paper {paper_id} not found in OpenSearch - cannot search for similar papers")
            except Exception as e:
                logger.warning(f"Error retrieving similar paper errors: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # static checks first before AI
        static_issues = self._check_static_issues(current_code, dataset_name)
        if static_issues:
            logger.info(f"Found {len(static_issues)} static issues, applying quick fixes...")
            current_code, quick_fixes = self._apply_quick_fixes(current_code, static_issues, dataset_name)
            fixes_applied.extend(quick_fixes)
        
        execution_tests = []  # Track execution test results
        
        # AI analysis 
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Code review iteration {iteration}/{self.max_iterations}")
            
            # If execution testing is enabled, test the code on Trainium first
            execution_errors = []
            if self.enable_execution_testing and self.trainium_endpoint and paper_id:
                logger.info(f"🧪 Testing code on Trainium (iteration {iteration}) with {self.stability_time}s stability check...")
                exec_result = self._test_code_on_trainium(
                    current_code, paper_id, paper_title, iteration
                )
                execution_tests.append(exec_result)
                
                # Check if code ran for stability_time without errors
                exec_time = exec_result.get('execution_time', 0)
                if exec_result.get('success') and exec_time >= self.stability_time:
                    logger.info(f"✅ Code executed successfully for {exec_time}s (>= {self.stability_time}s stability threshold) - code is stable!")
                    # Code is stable - stop reviewing
                    break
                elif exec_result.get('success') and exec_time < self.stability_time:
                    logger.info(f"⚠️ Code executed but only ran for {exec_time}s (< {self.stability_time}s) - continuing review...")
                    # Code runs but didn't reach stability threshold - continue reviewing
                else:
                    # Extract errors from execution
                    execution_errors = self._extract_execution_errors(exec_result)
                    if execution_errors:
                        logger.warning(f"❌ Execution failed with {len(execution_errors)} errors")
                        for error in execution_errors[:3]:
                            logger.warning(f"   - {error[:100]}")
            
            # In first iteration, include similar paper errors for proactive checking
            similar_errors_context = []
            if iteration == 1 and similar_paper_errors:  # iteration is 1-based in the loop (starts at 1)
                # Extract error messages from similar papers
                for similar_error in similar_paper_errors:
                    error_data = similar_error.get('error_data', {})
                    error_msg = error_data.get('error_message', '') or error_data.get('stderr', '')[:200]
                    if error_msg:
                        similar_paper_id = similar_error.get('paper_id', 'unknown')
                        similar_errors_context.append(f"Similar paper error ({similar_paper_id}): {error_msg[:150]}")
                
                if similar_errors_context:
                    logger.info(f"📋 Including {len(similar_errors_context)} errors from similar papers for proactive checking")
            
            # Analyze code for issues (using current_code which may have been fixed in previous iteration)
            logger.debug(f"Iteration {iteration}: Analyzing code ({len(current_code)} chars)")
            issues = self._analyze_code_with_ai(
                current_code, dataset_name, fixes_applied, execution_errors, similar_errors_context
            )
            
            # If we have execution errors, add them to the issues
            if execution_errors:
                if not issues:
                    issues = {"issues": [], "fixes_needed": []}
                issues["issues"] = execution_errors + issues.get("issues", [])
                issues["fixes_needed"] = [
                    f"Fix execution error: {err}" for err in execution_errors[:3]
                ] + issues.get("fixes_needed", [])
            
            # If we have similar paper errors in first iteration, add them to issues
            if iteration == 1 and similar_errors_context:
                if not issues:
                    issues = {"issues": [], "fixes_needed": []}
                # Add similar paper errors as proactive warnings
                issues["issues"] = similar_errors_context + issues.get("issues", [])
                issues["fixes_needed"] = [
                    f"Proactively check for: {err.split(': ', 1)[1] if ': ' in err else err}" 
                    for err in similar_errors_context[:3]
                ] + issues.get("fixes_needed", [])
            
            if not issues or issues.get('no_issues', False):
                # If execution testing is enabled and we haven't tested yet, test first
                if self.enable_execution_testing and self.trainium_endpoint and paper_id:
                    if not execution_tests or not any(t.get('success') for t in execution_tests):
                        logger.info("⚠️ AI found no issues, but execution hasn't succeeded yet - continuing...")
                    else:
                        logger.info("✅ No issues found and execution succeeded - code is ready!")
                        break
                else:
                    logger.info("✅ No issues found - code is ready!")
                    break
            
            # Record issues found (even if fix fails, we want to track what was found)
            issues_found = issues.get('issues', [])
            fixes_needed = issues.get('fixes_needed', [])
            
            if issues_found:
                logger.info(f"Found {len(issues_found)} issues: {', '.join(issues_found[:3])}...")
            
            # Fix issues (using current_code - the fixed code from previous iteration if any)
            logger.debug(f"Iteration {iteration}: Fixing code ({len(current_code)} chars)")
            fixed_code = self._fix_code_with_ai(current_code, issues, dataset_name, fixes_applied, iteration)
            
            if fixed_code and fixed_code != current_code:
                fixes_applied.append({
                    'iteration': iteration,
                    'issues_found': issues_found,
                    'fixes': fixes_needed,  # Use fixes_needed, not fixes
                    'execution_errors': execution_errors  # Store execution errors for next iteration
                })
                # Log code changes for debugging
                logger.info(f"Code length changed: {len(current_code)} -> {len(fixed_code)} chars")
                # CRITICAL: Update current_code so next iteration uses the fixed code
                current_code = fixed_code
                logger.info(f"✅ Applied fixes in iteration {iteration} - code updated (next iteration will use this fixed code)")
            elif fixed_code == current_code:
                # Fix returned same code - might mean no changes needed or fix failed
                logger.warning(f"⚠️ Fix returned same code - no changes applied")
                logger.warning(f"   This usually means:")
                logger.warning(f"   1. AI extracted the original code block instead of the fixed one")
                logger.warning(f"   2. AI didn't understand how to fix the issues")
                logger.warning(f"   3. AI response format was incorrect")
                if issues_found:
                    logger.warning(f"   Issues that should have been fixed: {', '.join(issues_found[:3])}...")
                # Still record issues found and execution errors
                fixes_applied.append({
                    'iteration': iteration,
                    'issues_found': issues_found,
                    'fixes': fixes_needed,
                    'execution_errors': execution_errors,  # Store execution errors even if fix failed
                    'no_changes': True  # Mark that no changes were made
                })
                # If we have issues but AI didn't fix them, break to avoid infinite loop
                if issues_found or execution_errors:
                    logger.error("❌ Code fix failed - AI returned identical code despite issues found")
                    break
                else:
                    # No issues found, so same code is fine
                    break
            elif fixed_code is None:
                # Fix failed - record issues but warn
                logger.error(f"❌ Code fix failed - AI could not generate fixed code")
                logger.error(f"   Issues found: {len(issues_found)}")
                # Still record the issues and execution errors so we know what was wrong
                fixes_applied.append({
                    'iteration': iteration,
                    'issues_found': issues_found,
                    'fixes': fixes_needed,
                    'execution_errors': execution_errors,  # Store execution errors even if fix failed
                    'fix_failed': True  # Mark that fix attempt failed
                })
                # Don't break - try to continue with original code, but log the failure
                logger.warning("⚠️ Continuing with original code (fix failed)")
                break
            else:
                # Fix returned same code - might mean no changes needed or fix failed
                logger.warning(f"⚠️ Fix returned same code - no changes applied")
                # Still record issues found and execution errors
                fixes_applied.append({
                    'iteration': iteration,
                    'issues_found': issues_found,
                    'fixes': fixes_needed,
                    'execution_errors': execution_errors,  # Store execution errors even if fix failed
                    'no_changes': True  # Mark that no changes were made
                })
                break
        
        review_time = time.time() - review_start_time
        logger.info(f"Code review completed in {review_time:.2f}s ({iteration} iterations)")
        
        return {
            'code': current_code,
            'fixes_applied': fixes_applied,
            'iterations': iteration,
            'review_time': review_time,
            'success': len(fixes_applied) > 0 or iteration == 1,
            'execution_tests': execution_tests if self.enable_execution_testing else []
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
            
            # Check for invalid xm.optimizer usage (CRITICAL - common error)
            if 'xm.optimizer' in code:
                issues.append("CRITICAL: xm.optimizer does NOT exist (e.g., xm.optimizer.SGD is WRONG). Use torch.optim.SGD instead, then use xm.optimizer_step(optimizer) instead of optimizer.step()")
            
            # Check for other non-existent xm APIs (CRITICAL - AI hallucination)
            if 'xm.xla_device_context' in code:
                issues.append("CRITICAL: xm.xla_device_context() does NOT exist. Use `device = xm.xla_device()` and `model.to(device)` instead. Do NOT use context managers with xm APIs.")
            if 'xm.mark_step_context' in code:
                issues.append("CRITICAL: xm.mark_step_context() does NOT exist. Just call `xm.mark_step()` directly, no context manager needed.")
            if 'xm.send_cpu_data_to_device' in code:
                issues.append("CRITICAL: xm.send_cpu_data_to_device() does NOT exist. Use `.to(device)` instead (e.g., `criterion.to(device)` or `model.to(device)`).")
            if 'xm.XlaModule' in code:
                issues.append("CRITICAL: xm.XlaModule does NOT exist. Use regular `nn.Module` for all classes. XLA compatibility comes from using XLA device and operations, NOT from inheriting from a special base class.")
            if 'xm.dot_general' in code:
                issues.append("CRITICAL: xm.dot_general() does NOT exist. Use regular PyTorch operations like `torch.matmul()` or `torch.mm()` - these ARE supported in XLA.")
            if 'xm.dot(' in code:
                issues.append("CRITICAL: xm.dot() does NOT exist. Use regular PyTorch operations like `torch.matmul()` or `torch.mm()` - these ARE supported in XLA.")
            if 'xm.scalar_tensor_to_python_scalar' in code:
                issues.append("CRITICAL: xm.scalar_tensor_to_python_scalar() does NOT exist. Use `.item()` on scalar tensors or `int()` for size conversions.")
            if 'xm.tensor(' in code:
                issues.append("CRITICAL: xm.tensor() does NOT exist. Use `torch.tensor()` instead. XLA tensors are created by moving regular PyTorch tensors to XLA device.")
            if 'xm.optimizer_step(optimizer, sync=' in code or 'xm.optimizer_step(optimizer, sync=True' in code or 'xm.optimizer_step(optimizer, sync=False' in code:
                issues.append("CRITICAL: xm.optimizer_step() does NOT accept a 'sync' parameter. Use `xm.optimizer_step(optimizer)` without any sync parameter.")
            if 'xm.xla_device_context' in code:
                issues.append("CRITICAL: xm.xla_device_context() does NOT exist. Remove the context manager and use direct calls: `device = xm.xla_device()` and `model.to(device)`.")
            if 'xm.mark_step_context' in code:
                issues.append("CRITICAL: xm.mark_step_context() does NOT exist. Just call `xm.mark_step()` directly, no context manager needed.")
            
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
        
        # Check for shape mismatches in normalization layers (common with vision + transformer)
        if 'LayerNorm' in code or 'BatchNorm' in code:
            # Check if using vision dataset (MNIST, CIFAR) with transformer architecture
            vision_datasets = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']
            is_vision = any(dataset in code.lower() for dataset in vision_datasets)
            has_transformer = 'Transformer' in code or 'MultiheadAttention' in code or 'encoder' in code.lower()
            
            if is_vision and has_transformer:
                # Check if input is properly processed before normalization
                # Should have: flatten/view → projection → reshape before LayerNorm
                has_flatten = 'flatten' in code.lower() or '.view(' in code or '.reshape(' in code
                has_projection = 'Linear' in code and ('projection' in code.lower() or 'embed' in code.lower() or 'input_proj' in code.lower())
                
                # Check if LayerNorm is applied directly to raw input (wrong)
                # Pattern: model(inputs) where inputs is [batch, channels, H, W] and model has LayerNorm expecting [batch, seq, d_model]
                if not has_projection or (has_transformer and not has_flatten):
                    issues.append("CRITICAL: Vision dataset (MNIST/CIFAR) used with Transformer but input not properly processed. Need to flatten image → project to d_model → reshape to [batch, seq_len, d_model] before LayerNorm. Current code likely passes raw images [batch, 1, 28, 28] to LayerNorm expecting [batch, seq, 512].")
        
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
        
        # Check for ellipsis (...) placeholders (CRITICAL - incomplete code)
        # Pattern: variable = ... (with optional comment)
        # Match: variable_name = ... (with optional whitespace and comment)
        ellipsis_pattern = r'(\w+)\s*=\s*\.\.\.(?:\s*#.*)?'
        matches = re.findall(ellipsis_pattern, code, re.MULTILINE)
        seen_vars = set()
        for var_name in matches:
            if var_name not in seen_vars:
                seen_vars.add(var_name)
                issues.append(f"CRITICAL: Variable '{var_name}' is assigned to ellipsis (...) placeholder - code is incomplete. Must initialize with actual value (e.g., model, tensor, or proper object).")
        
        # Check for LoRA/Adapter layer issues (CRITICAL - common mistakes)
        if 'lora' in code.lower() or 'adapter' in code.lower() or 'LoRA' in code:
            # Check for wrong layer traversal pattern
            if 'zip(lora_layers' in code or 'zip(self.lora_layers' in code:
                if 'self.layers' not in code or 'nn.ModuleList' not in code:
                    issues.append("CRITICAL: LoRA code uses zip(lora_layers, model) which misaligns layers. Use unified self.layers = nn.ModuleList() that includes both LoRA-wrapped and original layers.")
            
            # Check for wrong LoRA math (inputs + matmul pattern)
            if 'inputs + torch.matmul(inputs' in code or 'x + torch.matmul(x' in code:
                if 'F.linear' not in code or 'self.layer.weight +' not in code:
                    issues.append("CRITICAL: LoRA forward pass uses wrong pattern 'inputs + torch.matmul(inputs, A) @ B.T'. Should use 'F.linear(x, self.layer.weight + self.B @ self.A, self.layer.bias)' (W' = W + B@A formula).")
            
            # Check for wrong LoRA formula (B@A.T instead of B@A)
            if 'self.B @ self.A.T' in code or 'torch.matmul(self.B, self.A.T)' in code or 'matmul(self.B, self.A.T)' in code:
                issues.append("CRITICAL: LoRA formula uses wrong transpose. Should be 'self.B @ self.A' (W' = W + B@A), NOT 'self.B @ self.A.T' or 'torch.matmul(self.B, self.A.T)'.")
            
            # Check for wrapping custom models with complex forward()
            if 'LoRAModel(' in code:
                # Check if LoRAModel is wrapping a custom class instance (not nn.Sequential)
                # Pattern: LoRAModel(SomeModelClass(), ...) or LoRAModel(SomeModel(), ...)
                lora_pattern = r'LoRAModel\s*\(\s*(\w+)\s*\([^)]*\)'
                matches = re.findall(lora_pattern, code)
                for match in matches:
                    # If it's wrapping a class instantiation (not nn.Sequential)
                    if match != 'nn.Sequential' and match[0].isupper():  # Class name starts with uppercase
                        # Check if that class has a custom forward method
                        class_pattern = rf'class\s+{match}\s*\([^)]*\):.*?def\s+forward\s*\('
                        if re.search(class_pattern, code, re.DOTALL):
                            issues.append(f"CRITICAL: LoRAModel is wrapping custom model '{match}' with complex forward() method. LoRAModel can only wrap nn.Sequential or simple layer lists, not models with custom forward logic (Conv2d, flatten, etc.). Use nn.Sequential for base model instead.")
                            break
            
            # Check for wrong parameter dimensions
            if 'torch.zeros(rank, layer.weight.shape[1]' in code:
                issues.append("CRITICAL: LoRA parameter A has wrong dimensions. Should be 'torch.zeros(rank, layer.in_features)' not 'torch.zeros(rank, layer.weight.shape[1])'.")
            if 'torch.zeros(layer.weight.shape[0], rank' in code:
                issues.append("CRITICAL: LoRA parameter B has wrong dimensions. Should be 'torch.zeros(layer.out_features, rank)' not 'torch.zeros(layer.weight.shape[0], rank)'.")
            
            # Check for tensor shape mismatches in LoRA operations
            if 'self.B @ self.A.T' in code or 'torch.matmul(self.B, self.A.T' in code:
                issues.append("CRITICAL: LoRA matrix multiplication uses wrong transpose. Should be 'self.B @ self.A' (produces [out_features, in_features]), not 'self.B @ self.A.T' (produces wrong shape). This will cause XLA 'Check failed: dim1 == dim2' error.")
            if 'self.A @ self.B' in code or 'torch.matmul(self.A, self.B' in code:
                issues.append("CRITICAL: LoRA matrix multiplication has wrong order. Should be 'self.B @ self.A' (produces [out_features, in_features] to match weight matrix), not 'self.A @ self.B' (produces wrong shape). This will cause XLA 'Check failed: dim1 == dim2' error.")
            
            # Check for using weight.shape instead of in_features/out_features
            if 'layer.weight.shape[0]' in code or 'layer.weight.shape[1]' in code:
                # Only flag if it's in LoRA context (parameter initialization or forward pass)
                if 'LoRA' in code or 'lora' in code.lower() or 'self.A' in code or 'self.B' in code:
                    issues.append("CRITICAL: Using 'layer.weight.shape[...]' instead of 'layer.in_features'/'layer.out_features' in LoRA. This can cause shape mismatches. Use 'layer.in_features' and 'layer.out_features' instead.")
            
            # Check for adding tensors without shape verification
            if 'self.layer.weight +' in code and ('self.B @ self.A' in code or 'self.A @ self.B' in code or 'torch.matmul' in code):
                # Verify the operation is correct
                if 'self.B @ self.A' not in code and 'torch.matmul(self.B, self.A' not in code:
                    issues.append("CRITICAL: Adding weight matrix to LoRA matrices but operation may have wrong shape. Ensure 'B @ A' produces [out_features, in_features] to match weight matrix shape. Wrong shapes will cause XLA 'Check failed: dim1 == dim2' error.")
            
            # Check for reassigning layer.weight in forward pass (CRITICAL XLA error)
            if 'self.layer.weight =' in code or 'layer.weight =' in code:
                # Check if it's in a forward method
                forward_pattern = r'def\s+forward\s*\([^)]*\):.*?(?:self\.layer\.weight\s*=|layer\.weight\s*=)'
                if re.search(forward_pattern, code, re.DOTALL):
                    issues.append("CRITICAL: Reassigning layer.weight in forward() method. This creates a new Parameter each forward pass, breaks XLA, and causes optimizer to lose track of parameters. Use F.linear() with modified weight instead: 'return F.linear(x, self.original_weight + self.B @ self.A, self.layer.bias)'")
            
            # Check for creating new Parameter in forward pass
            if 'nn.Parameter(' in code:
                # Check if Parameter is created inside forward method
                forward_with_param_pattern = r'def\s+forward\s*\([^)]*\):.*?nn\.Parameter\s*\('
                if re.search(forward_with_param_pattern, code, re.DOTALL):
                    issues.append("CRITICAL: Creating nn.Parameter inside forward() method. Parameters must be created in __init__, not forward(). Creating new Parameters in forward() breaks XLA and causes infinite loops/timeouts. Use F.linear() or functional operations instead.")
            
            # Check for manual gradient adjustment
            if 'adjust_gradients' in code or 'A.grad.data =' in code or 'B.grad.data =' in code:
                issues.append("CRITICAL: LoRA code has manual gradient adjustment function. LoRA gradients propagate automatically - remove adjust_gradients() function and all manual gradient manipulation.")
            
            # Check for .to(device) on loss function
            if 'CrossEntropyLoss().to(device)' in code or 'MSELoss().to(device)' in code:
                issues.append("CRITICAL: Loss function has .to(device) - loss functions don't have parameters. Use 'criterion = nn.CrossEntropyLoss()' without .to(device).")
            
            # Check for missing model.train()
            if 'for epoch in range' in code and 'model.train()' not in code:
                issues.append("CRITICAL: Training loop missing model.train() at start of each epoch. Add 'model.train()' before training loop to ensure dropout/batchnorm work correctly.")
        
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
    
    def _test_code_on_trainium(self, code: str, paper_id: str, paper_title: Optional[str], 
                                iteration: int) -> Dict[str, Any]:
        """
        Execute code on Trainium and return results.
        
        Args:
            code: Code to execute
            paper_id: Paper ID
            paper_title: Paper title (optional)
            iteration: Current iteration number
            
        Returns:
            Execution result dictionary
        """
        if not self.trainium_endpoint:
            return {
                "success": False,
                "error": "Trainium endpoint not configured",
                "iteration": iteration
            }
        
        # Use stability_time + buffer for the test (we want to see if it runs for stability_time)
        test_timeout = self.stability_time + 30  # Add 30s buffer
        endpoint = f"{self.trainium_endpoint}/execute"
        payload = {
            "paper_id": f"{paper_id}_review_iter_{iteration}",
            "code": code,
            "timeout": test_timeout
        }
        
        if paper_title:
            payload["paper_title"] = paper_title
        
        try:
            logger.info(f"Sending code to Trainium at {endpoint} (timeout: {self.execution_timeout}s)")
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.execution_timeout + 60  # Add buffer for HTTP overhead
            )
            response.raise_for_status()
            result = response.json()
            result["iteration"] = iteration
            return result
        except requests.exceptions.Timeout:
            logger.warning(f"Execution test timed out after {self.execution_timeout}s")
            # Timeout during review means code is hanging or taking too long - this is a real issue
            return {
                "success": False,
                "error_message": f"Code execution timed out after {self.execution_timeout}s - code may be hanging, have infinite loop, or Neuron compilation is taking too long (should complete in <2min for syntax/import checks)",
                "error_type": "timeout",
                "stderr": f"Execution timed out after {self.execution_timeout}s. This suggests:\n1. Code has infinite loop or blocking operation\n2. Neuron compilation is taking too long (possible compilation error)\n3. Code is waiting for input or network\n\nFor code review, code should fail fast with clear errors, not hang.",
                "iteration": iteration
            }
        except Exception as e:
            logger.error(f"Error executing code on Trainium: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "error_type": "execution_error",
                "iteration": iteration
            }
    
    def _extract_execution_errors(self, exec_result: Dict[str, Any]) -> List[str]:
        """
        Extract error messages from execution result.
        
        Args:
            exec_result: Execution result dictionary
            
        Returns:
            List of error message strings
        """
        errors = []
        
        if exec_result.get('success'):
            return errors
        
        # Check for timeout errors first (these are important - code is hanging)
        if exec_result.get('error_type') == 'timeout':
            timeout_msg = exec_result.get('error_message', 'Execution timed out')
            # For code review, timeouts indicate real problems:
            # - Code has infinite loop
            # - Neuron compilation hanging (possible compilation error)
            # - Code waiting for input/network
            errors.append(f"EXECUTION ERROR: {timeout_msg}")
            # Add helpful context
            if 'stderr' in exec_result and exec_result['stderr']:
                errors.append(f"EXECUTION ERROR: Code execution timed out - this suggests code has problems (infinite loop, hanging compilation, or blocking operation)")
            return errors[:5]  # Return timeout error immediately
        
        # Extract from stderr
        stderr = exec_result.get('stderr', '')
        if stderr:
            # Look for Python tracebacks - capture full error message
            # Pattern: Traceback... followed by File lines, ending with ErrorType: message
            traceback_pattern = r'Traceback \(most recent call last\):(.*?)((?:AttributeError|TypeError|ValueError|RuntimeError|ImportError|KeyError|IndexError|NameError|ModuleNotFoundError):\s*.+?)(?=\n\n|\ntime=|$)'
            matches = re.findall(traceback_pattern, stderr, re.DOTALL)
            for match in matches:
                # match[0] is the traceback body, match[1] is the error line
                error_line = match[1].strip()
                if error_line:
                    # Get the last few lines of context from traceback for better understanding
                    traceback_body = match[0].strip()
                    traceback_lines = [l.strip() for l in traceback_body.split('\n') if l.strip() and 'File "' in l]
                    if traceback_lines:
                        # Prefer user code files (not site-packages or internal PyTorch files)
                        user_code_lines = [l for l in traceback_lines if '/site-packages/' not in l and '/torch/nn/' not in l]
                        if user_code_lines:
                            # Use the last user code line (most relevant)
                            last_file_line = user_code_lines[-1]
                        else:
                            # Fall back to last line if no user code found
                            last_file_line = traceback_lines[-1]
                        
                        # Extract line number and code if available
                        file_match = re.search(r'File ".*?", line (\d+), in (.+)', last_file_line)
                        if file_match:
                            line_num = file_match.group(1)
                            func_name = file_match.group(2)
                            # Also try to get the actual code line from traceback
                            code_line = ""
                            for line in traceback_body.split('\n'):
                                if line.strip() and not line.strip().startswith('File ') and not line.strip().startswith('Traceback'):
                                    code_line = line.strip()
                                    break
                            if code_line:
                                errors.append(f"EXECUTION ERROR: {error_line} (at line {line_num} in {func_name}: {code_line})")
                            else:
                                errors.append(f"EXECUTION ERROR: {error_line} (at line {line_num} in {func_name})")
                        else:
                            errors.append(f"EXECUTION ERROR: {error_line}")
                    else:
                        errors.append(f"EXECUTION ERROR: {error_line}")
            
            # If no traceback found, look for error patterns directly
            if not errors:
                error_patterns = [
                    r'(AttributeError|TypeError|ValueError|RuntimeError|ImportError|KeyError|IndexError|NameError|ModuleNotFoundError):\s*(.+?)(?=\n|$)',
                ]
                for pattern in error_patterns:
                    matches = re.findall(pattern, stderr, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        if isinstance(match, tuple):
                            error_type = match[0]
                            error_msg = match[1].strip()
                            # Truncate very long messages
                            if len(error_msg) > 200:
                                error_msg = error_msg[:200] + "..."
                            errors.append(f"EXECUTION ERROR: {error_type}: {error_msg}")
                        else:
                            error_msg = match.strip()
                            if len(error_msg) > 200:
                                error_msg = error_msg[:200] + "..."
                            errors.append(f"EXECUTION ERROR: {error_msg}")
        
        # Extract from error_message if available
        error_message = exec_result.get('error_message')
        if error_message and error_message not in errors:
            errors.append(f"EXECUTION ERROR: {error_message}")
        
        # If no specific errors found, add generic error
        if not errors:
            return_code = exec_result.get('return_code', -1)
            errors.append(f"EXECUTION ERROR: Code execution failed (return code: {return_code})")
        
        return errors[:5]  # Limit to 5 errors
    
    def _analyze_code_with_ai(self, code: str, dataset_name: str = None, 
                              previous_fixes: List[Dict] = None,
                              execution_errors: List[str] = None,
                              similar_paper_errors: List[str] = None) -> Dict[str, Any]:
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
        
        # Use .format() instead of f-string to avoid issues with curly braces in code
        # Escape JSON braces in the template, then format with actual values
        trainium_errors_ref = self._get_trainium_error_reference()
        
        # Add similar paper errors to prompt (proactive checking)
        similar_errors_text = ""
        if similar_paper_errors:
            similar_errors_text = "\n\n**🔍 PROACTIVE ERROR CHECKING - ERRORS FROM SIMILAR PAPERS:**\n"
            similar_errors_text += "The following errors occurred in similar papers when running on Trainium. "
            similar_errors_text += "Review the code carefully to ensure these errors will NOT occur:\n"
            for i, err in enumerate(similar_paper_errors, 1):
                similar_errors_text += f"{i}. {err}\n"
            similar_errors_text += "\nThese are errors from papers with similar abstracts. "
            similar_errors_text += "Check your code proactively to prevent these same errors.\n"
        
        # Add execution errors to prompt if available
        execution_errors_text = ""
        if execution_errors:
            execution_errors_text = "\n\n**⚠️ CRITICAL: REAL EXECUTION ERRORS FROM TRAINIUM:**\n"
            execution_errors_text += "The following errors occurred when this code was executed on Trainium:\n"
            for i, err in enumerate(execution_errors, 1):
                execution_errors_text += f"{i}. {err}\n"
            execution_errors_text += "\nYou MUST fix these specific errors. These are REAL runtime errors, not hypothetical issues.\n"
        
        analysis_prompt_template = """You are an expert code reviewer for PyTorch code that will run on AWS Trainium using the Neuron SDK (XLA devices).

Your task: Review this PyTorch code and identify ANY issues that would prevent it from running correctly on Trainium with the Neuron SDK. Look for ALL types of errors, not just common ones.

CRITICAL: This code MUST use the Neuron SDK (torch_xla) for Trainium execution. Ensure all XLA operations are correct.

```python
{code}
```

Dataset being used: {dataset_name}

{fix_history}

{execution_errors}

COMPREHENSIVE REVIEW CHECKLIST - Check for ALL of these:

{trainium_errors}

1. **Neuron SDK / XLA / Trainium Compatibility (CRITICAL):**
   - MUST use torch_xla.core.xla_model as xm (Neuron SDK XLA module)
   - MUST use xm.xla_device() to get Trainium device (NOT torch.device('cuda') or 'cpu')
   - MUST use xm.optimizer_step(optimizer) instead of optimizer.step() (Neuron SDK requirement)
   - MUST call xm.mark_step() after each backward pass (Neuron SDK synchronization)
   - All tensor operations compatible with XLA (no in-place operations on indexed tensors)
   - Tensor size operations: tensor.size(0) returns a tensor in XLA, must use int() for arithmetic
   - No CUDA-specific code (.cuda(), device='cuda', torch.device('cuda'))
   - Ensure all tensors moved to XLA device before operations
   
   **IMPORTANT: DO NOT incorrectly flag regular PyTorch operations as incompatible:**
   - `torch.matmul()`, `torch.mm()`, `nn.Linear`, `nn.Conv2d`, etc. ARE compatible with XLA
   - These operations work fine in XLA - compatibility comes from device placement, not special APIs
   - Only flag issues if code uses CUDA-specific operations or non-existent xm.* APIs
   
   **VALID torch_xla.core.xla_model (xm) APIs ONLY:**
   - xm.xla_device() - Get XLA device
   - xm.optimizer_step(optimizer) - XLA optimizer step (NO sync parameter - just xm.optimizer_step(optimizer))
   - xm.mark_step() - Synchronize XLA computation
   - xm.rendezvous(tag) - Synchronization barrier (requires tag string)
   - xm.get_ordinal() - Get device ordinal (distributed)
   - xm.get_world_size() - Get world size (distributed)
   
   **CRITICAL: Regular PyTorch operations ARE supported in XLA:**
   - `torch.matmul()`, `torch.mm()`, `torch.add()`, `torch.mul()`, etc. - ALL work in XLA
   - `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, etc. - ALL standard PyTorch modules work in XLA
   - `nn.Module` - Use regular `nn.Module` for all classes, NOT `xm.XlaModule` (which doesn't exist)
   - XLA compatibility comes from using XLA device (`xm.xla_device()`) and XLA optimizer step, NOT from special APIs
   
   **DO NOT suggest or use non-existent APIs like:**
   - xm.optimizer - THIS DOES NOT EXIST (e.g., xm.optimizer.SGD is WRONG)
   - xm.XlaModule - THIS DOES NOT EXIST (use regular `nn.Module`)
   - xm.dot() or xm.dot_general() - THESE DO NOT EXIST (use `torch.matmul()` or `torch.mm()`)
   - xm.tensor() - THIS DOES NOT EXIST (use `torch.tensor()`)
   - xm.scalar_tensor_to_python_scalar() - THIS DOES NOT EXIST (use `.item()` or `int()`)
   - xm.xla_device_context() - THIS DOES NOT EXIST (use `device = xm.xla_device()` and `model.to(device)` instead)
   - xm.mark_step_context() - THIS DOES NOT EXIST (just call `xm.mark_step()` directly, no context manager)
   - xm.send_cpu_data_to_device() - THIS DOES NOT EXIST (use `.to(device)` instead)
   - xm.save_memory_state() - THIS DOES NOT EXIST
   - xm.optimizer_step(optimizer, sync=True) - sync parameter DOES NOT EXIST (just use xm.optimizer_step(optimizer))
   - Any other xm.* functions not listed above
   - Only suggest fixes using the VALID APIs listed above
   
   **OPTIMIZER USAGE (CRITICAL):**
   - WRONG: `optimizer = xm.optimizer.SGD(...)` - xm.optimizer does NOT exist
   - CORRECT: `optimizer = torch.optim.SGD(...)` then use `xm.optimizer_step(optimizer)` instead of `optimizer.step()`
   - Use regular PyTorch optimizers (torch.optim.SGD, torch.optim.Adam, etc.) - NOT xm.optimizer.*

2. **Data Handling (CRITICAL):**
   - MUST use `from dataset_loader import load_dataset` - DO NOT use torchvision.datasets
   - WRONG: `import torchvision.datasets as datasets` or `datasets.MNIST(...)`
   - WRONG: `train_loader.to(device)` or `test_loader.to(device)` - DataLoaders CANNOT be moved to device
   - CORRECT: `train_loader, test_loader = load_dataset('mnist', batch_size=128)`
   - CORRECT: Move tensors to device INSIDE the training loop: `inputs = inputs.to(device)`, NOT the DataLoader
   - IMDB dataset: Returns (text_strings, labels) - text must be tokenized before use
   - DataLoader iteration: Handle both (tensor, tensor) and (text_strings, labels) formats
   - All tensors moved to device before operations (but NOT DataLoaders)
   - Proper handling of batch data unpacking

3. **Model Output Handling:**
   - Model may return tuples - check isinstance() before using
   - Proper unpacking of model outputs
   - Handle optional return values (e.g., return_bias_scores=True)

4. **Type Errors:**
   - Mixing tensors with Python ints/floats in arithmetic
   - Calling methods on wrong types (e.g., .to() on list, .item() on non-scalar)
   - WRONG: `train_loader.to(device)` - DataLoaders are NOT tensors and cannot be moved to device
   - CORRECT: Move tensors from DataLoader to device: `inputs, labels = inputs.to(device), labels.to(device)`
   - Indexing issues with XLA tensors
   
5. **nn.ModuleDict Key Errors (CRITICAL):**
   - WRONG: `nn.ModuleDict({{'1.weight': module}})` - keys cannot contain dots
   - WRONG: `nn.ModuleDict({{f'{{name}}.weight': module}})` - if name contains dots or creates keys with dots
   - CORRECT: Replace dots with underscores: `nn.ModuleDict({{name.replace('.', '_'): module}})`
   - CORRECT: Use a different naming scheme without dots: `nn.ModuleDict({{f'layer_{{i}}': module for i, module in enumerate(...)}})`
   - PyTorch/XLA requires ModuleDict keys to be valid Python identifiers (no dots)

6. **Import Errors:**
   - All used modules imported (math, random, collections, etc.)
   - Correct import paths

7. **Logic Errors:**
   - Division by zero or None
   - Uninitialized variables
   - **CRITICAL: Ellipsis (...) placeholders (incomplete code):**
     - WRONG: `base_model = ...` or `model = ...` - ellipsis is NOT a valid model/tensor/object
     - WRONG: `variable = ... # Initialize here` - this is a placeholder, not actual initialization
     - CORRECT: Must initialize with actual value (e.g., `base_model = nn.Sequential(...)` or `model = MyModel()`)
     - Check for ANY variable assignment to `...` - this will cause AttributeError at runtime
   - Incorrect tensor shapes/dimensions
   - Wrong device placement
   - **Shape Mismatch in Normalization Layers (CRITICAL):**
     - LayerNorm expects input shape `[batch, ..., normalized_shape]` where last dimension matches normalized_shape
     - WRONG: Applying LayerNorm(d_model) to image tensor `[batch, channels, height, width]` without flattening/projection
     - WRONG: For vision datasets (MNIST, CIFAR), passing raw images `[batch, 1, 28, 28]` directly to transformer expecting `[batch, seq_len, d_model]`
     - CORRECT: For vision + transformer: flatten image → project to d_model → reshape to `[batch, seq_len, d_model]` → then apply LayerNorm
     - CORRECT: For vision + transformer: `x = x.view(batch_size, -1)` then `x = self.projection(x)` then `x = x.view(batch_size, seq_len, d_model)` before normalization

8. **Runtime Errors:**
   - AttributeError (calling methods on wrong types)
     - **CRITICAL: AttributeError: 'ellipsis' object has no attribute 'X'** - Variable assigned to `...` placeholder
     - Example: `base_model = ...` then `base_model.named_modules()` → AttributeError
   - TypeError (wrong argument types)
   - ValueError (wrong tensor shapes, dimensions)
   - IndexError (out of bounds access)
   - KeyError: 'module name can\'t contain "."' - ModuleDict keys with dots
   - RuntimeError: "Given normalized_shape=[X], expected input with shape [*, X], but got input of size[...]" - Shape mismatch in LayerNorm/BatchNorm
   - RuntimeError: "Check failed: dim1 == dim2" - XLA tensor shape mismatch when adding/multiplying tensors (common in LoRA when B@A has wrong shape)

9. **XLA-Specific Gotchas:**
   - Using tensor.size(0) directly in arithmetic without int() conversion
   - Using tensor values in Python control flow without .item()
   - In-place operations that XLA doesn't support
   - Loss functions not moved to device (they shouldn't be)
   
10. **Gradient Access Errors (CRITICAL):**
   - Accessing .grad during forward pass - gradients are None until loss.backward() is called
   - WRONG: Using param.grad in forward() method or before backward()
   - WRONG: Multiplying Parameter * None (when .grad is None)
   - CORRECT: Only access .grad after loss.backward() in training loop
   - CORRECT: Use param.data or param directly in forward pass, not param.grad

11. **LoRA / Adaptation Layer Errors (CRITICAL - IF CODE USES LORA/ADAPTERS):**
   - **WRONG Layer Traversal:**
     - ❌ `for layer in model:` then `zip(lora_layers, model)` - This misaligns layers
     - ❌ Separate `lora_layers` list that doesn't match model layer count
     - ✅ CORRECT: Use unified `self.layers = nn.ModuleList()` that includes both LoRA-wrapped and original layers
   - **WRONG LoRA Math:**
     - ❌ `return inputs + torch.matmul(inputs, self.A) @ self.B.T` - Wrong dimensions and bypasses base layer
     - ❌ `torch.matmul(self.B, self.A.T)` - Wrong: should be `self.B @ self.A` (formula is W' = W + B@A, not W + B@A^T)
     - ❌ `self.A = nn.Parameter(torch.zeros(rank, layer.weight.shape[1]))` - Wrong: A should be `[rank, in_features]`
     - ❌ `self.B = nn.Parameter(torch.zeros(layer.weight.shape[0], rank))` - Wrong: B should be `[out_features, rank]`
     - ✅ CORRECT: `W' = W + B@A` where A is `[rank, in_features]`, B is `[out_features, rank]`
     - ✅ CORRECT: `return F.linear(x, self.layer.weight + self.B @ self.A, self.layer.bias)` (NOT self.B @ self.A.T)
   - **CRITICAL - Tensor Shape Mismatch (XLA Errors):**
     - ❌ RuntimeError: "Check failed: dim1 == dim2" - Adding tensors with incompatible shapes
     - ❌ Using `layer.weight.shape[0]` or `layer.weight.shape[1]` instead of `layer.in_features`/`layer.out_features`
     - ❌ `B @ A.T` or `A @ B` instead of `B @ A` - produces wrong shape [rank, rank] or [in_features, out_features] instead of [out_features, in_features]
     - ❌ Adding `W + B@A` where shapes don't match: W is [out_features, in_features] but B@A is wrong shape
     - ✅ VERIFY: `B @ A` where B is [out_features, rank] and A is [rank, in_features] produces [out_features, in_features] ✓
     - ✅ VERIFY: `W + (B @ A)` where both are [out_features, in_features] - shapes match ✓
     - ❌ WRONG: Any operation that adds/multiplies tensors without ensuring compatible shapes
     - ✅ CORRECT: Always use `layer.in_features` and `layer.out_features` for dimensions, never `layer.weight.shape[...]`
   - **CRITICAL - Parameter Reassignment in Forward Pass (XLA/Timeout Errors):**
     - ❌ `self.layer.weight = nn.Parameter(weight)` inside forward() - Creates new Parameter each forward pass
     - ❌ `layer.weight = nn.Parameter(...)` in forward() - Breaks XLA and optimizer parameter tracking
     - ❌ Creating `nn.Parameter()` inside forward() method - Parameters must be created in __init__
     - ✅ CORRECT: Use `F.linear(x, self.original_weight + self.B @ self.A, self.layer.bias)` instead
     - ✅ CORRECT: Compute modified weight as tensor, use functional operations (F.linear, F.conv2d, etc.)
     - ❌ WRONG: Reassigning layer.weight in forward() causes infinite loops, timeouts, and XLA errors
     - ✅ CORRECT: Store original weight in __init__, compute modified weight in forward(), use F.linear() to apply
   - **WRONG Model Wrapping:**
     - ❌ Wrapping custom models with complex forward() methods (e.g., `LoRAModel(ExampleModel())` where ExampleModel has Conv2d, flatten, etc.)
     - ❌ Using `model.children()` to iterate over custom model - breaks model structure
     - ✅ CORRECT: Only wrap `nn.Sequential` models or simple layer lists, not models with custom forward logic
   - **WRONG Gradient Adjustment:**
     - ❌ Manual `adjust_gradients()` function with incorrect matrix math
     - ❌ `A.grad.data = dL_dA - torch.matmul(dL_dA, B) * B.T` - Mixes matmul and elementwise, wrong shapes
     - ✅ CORRECT: Remove gradient adjustment - LoRA gradients propagate automatically through the decomposition
   - **WRONG Loss Function Device:**
     - ❌ `criterion = nn.CrossEntropyLoss().to(device)` - Loss functions don't have parameters
     - ✅ CORRECT: `criterion = nn.CrossEntropyLoss()` (no .to(device))
   - **WRONG Training Loop:**
     - ❌ Missing `model.train()` at start of each epoch - causes eval mode to persist
     - ❌ `int(labels.size(0))` on XLA tensors - should use `.size(0)` directly or `.sum().item()`
     - ✅ CORRECT: Call `model.train()` at start of each training epoch
     - ✅ CORRECT: Use `total += labels.size(0)` and `correct += (predicted == labels).sum().item()`
   - **Layer Alignment Issues:**
     - ❌ `for lora_layer, layer in zip(self.lora_layers, self.model):` - Stops early if counts don't match
     - ❌ Model has 4 layers (Flatten, Linear, ReLU, Linear) but lora_layers has only 2 (Linear layers)
     - ✅ CORRECT: Use single unified `self.layers` list and iterate: `for layer in self.layers: x = layer(x)`

IMPORTANT: Be thorough. Look for ANY error that would cause the code to fail, not just the common ones listed above. Think like a compiler/linter - catch everything.

Respond in JSON format:
{{
    "no_issues": false,
    "issues": ["detailed issue description 1", "detailed issue description 2", ...],
    "fixes_needed": ["specific fix instruction 1", "specific fix instruction 2", ...],
    "critical": true/false
}}

If no issues found, set "no_issues": true."""
        
        # Format the prompt with actual values (escaping braces in code)
        analysis_prompt = analysis_prompt_template.format(
            code=code.replace('{', '{{').replace('}', '}}'),
            dataset_name=dataset_name or 'unknown',
            fix_history=fix_history,
            execution_errors=similar_errors_text + execution_errors_text,  # Include similar paper errors first
            trainium_errors=trainium_errors_ref
        )

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
                          dataset_name: str = None, 
                          previous_fixes: List[Dict] = None,
                          iteration: int = 1) -> Optional[str]:
        """
        Use AI to fix code issues.
        
        Returns:
            Fixed code or None if fix failed
        """
        issues_list = issues.get('issues', [])
        fixes_needed = issues.get('fixes_needed', [])
        
        if not issues_list and not fixes_needed:
            return code
        
        # Build fix history context
        fix_history = ""
        if previous_fixes:
            fix_history = "\n⚠️ ITERATIVE FIXING CONTEXT - This is iteration {iteration}:\n"
            fix_history += "Previous fix attempts:\n"
            all_previous_execution_errors = []  # Collect all previous execution errors
            for fix in previous_fixes:
                prev_iter = fix.get('iteration', '?')
                prev_issues = fix.get('issues_found', [])
                prev_fixes = fix.get('fixes', [])
                prev_execution_errors = fix.get('execution_errors', [])
                prev_fixes_str = ', '.join(prev_fixes[:2]) if prev_fixes else 'none'
                fix_history += "  - Iteration {}: Found {} issues, attempted fixes: {}...\n".format(
                    prev_iter, len(prev_issues), prev_fixes_str
                )
                # Collect execution errors from previous iterations
                if prev_execution_errors:
                    all_previous_execution_errors.extend(prev_execution_errors)
                    fix_history += f"    Previous execution errors: {', '.join(prev_execution_errors[:2])}...\n"
            
            if all_previous_execution_errors:
                fix_history += "\n⚠️ PREVIOUS EXECUTION ERRORS (from earlier iterations):\n"
                for i, err in enumerate(all_previous_execution_errors[:5], 1):
                    fix_history += f"  {i}. {err}\n"
                fix_history += "These errors occurred in previous iterations. Make sure your fixes address them.\n"
            
            fix_history += "\nIMPORTANT: The code below is ALREADY PARTIALLY FIXED from previous iterations. "
            fix_history += "Some issues may persist because previous fixes were incomplete or incorrect. "
            fix_history += "You must provide BETTER fixes that actually resolve the remaining issues.\n"
        
        # Use .format() instead of f-string to avoid issues with curly braces in code
        issues_text = '\n'.join(f'- {issue}' for issue in issues_list)
        fixes_text = '\n'.join(f'- {fix}' for fix in fixes_needed)
        trainium_errors_ref = self._get_trainium_error_reference()
        
        fix_prompt_template = """Fix the following PyTorch code to work on AWS Trainium using the Neuron SDK (XLA devices).

CRITICAL: This code MUST use the Neuron SDK (torch_xla) for Trainium execution. All XLA operations must be correct.

{fix_history}

{trainium_errors}

Current issues identified (iteration {iteration}):
{issues_text}

{fixes_text}

Current code (this code has been through {iteration_minus_one} previous fix iteration(s)):
```python
{code}
```

Dataset: {dataset_name}

CRITICAL FIXES NEEDED:
1. Tokenizer input types (CRITICAL - COMMON BUG):
   - Tokenizers from transformers library expect STRINGS or LIST OF STRINGS, NOT tensors
   - WRONG: tokenizer(inputs, ...) where inputs is a tensor from dataloader
   - WRONG: tokenizer(list(inputs), ...) where inputs is a tensor (list() doesn't convert tensor to strings)
   - CORRECT: If inputs are strings, use tokenizer(inputs, ...) or tokenizer(list(inputs), ...)
   - CORRECT: If inputs are tensors from dataloader, you need to extract text strings first
   - For synthetic datasets: synthetic data returns tensors, NOT text - do NOT pass to tokenizer
   - For IMDB dataset: IMDB returns text strings - tokenize them in the training loop
   - For WikiText: WikiText returns text strings - tokenize them in the training loop
2. If using IMDB dataset: Add tokenization in training/test loops (IMDB returns text strings, not tokenized tensors)
3. Handle model tuple outputs: Check if model returns tuple and extract first element
4. XLA tensor size conversion (CRITICAL - MOST COMMON BUG):
   - Find ALL uses of tensor.size(0) or tensor.shape[0] in arithmetic operations
   - WRONG: count += inputs.size(0) or batch_size = inputs.size(0) or scores / count where count came from size(0)
   - CORRECT: count += int(inputs.size(0)) or batch_size = int(inputs.size(0))
   - Before division: count = int(count) if hasattr(count, 'item') else int(count) before scores / count
   - In XLA, tensor.size(0) returns a tensor, NOT a Python int - this causes AttributeError
5. Dataset loading (CRITICAL - COMMON BUG):
   - WRONG: Using `torchvision.datasets.MNIST(...)` or `import torchvision.datasets`
   - WRONG: `train_loader.to(device)` or `test_loader.to(device)` - DataLoaders cannot be moved to device
   - CORRECT: `from dataset_loader import load_dataset` then `train_loader, test_loader = load_dataset('mnist', batch_size=128)`
   - CORRECT: Move tensors to device in training loop: `inputs, labels = inputs.to(device), labels.to(device)`
   - If you see `torchvision.datasets` or `DataLoader.to(device)`, replace with dataset_loader and move tensors inside loop
6. nn.ModuleDict key errors (CRITICAL):
   - WRONG: `nn.ModuleDict({{f'{{name}}.weight': module}})` - keys cannot contain dots (will cause KeyError)
   - WRONG: `nn.ModuleDict({{'1.weight': module}})` - dots in keys are not allowed
   - CORRECT: Replace dots with underscores: `nn.ModuleDict({{name.replace('.', '_'): module}})`
   - CORRECT: Use index-based naming: `nn.ModuleDict({{f'layer_{{i}}': module for i, module in enumerate(...)}})`
   - PyTorch requires ModuleDict keys to be valid identifiers without dots
7. Ensure all tensors are moved to XLA device (xm.xla_device()) before operations (but NOT DataLoaders)
8. MUST use xm.optimizer_step(optimizer) instead of optimizer.step() (Neuron SDK requirement)
9. MUST call xm.mark_step() after each backward pass (Neuron SDK synchronization)
10. xm.rendezvous() MUST have a tag argument on Trainium/Neuron:
   - WRONG: xm.rendezvous()  (will cause IndexError: tuple index out of range)
   - CORRECT: xm.rendezvous('init') or xm.rendezvous('training') or any string tag
11. Gradient access errors (CRITICAL):
   - WRONG: Accessing param.grad in forward() method - gradients are None until loss.backward()
   - WRONG: Multiplying Parameter * None (e.g., self.alpha * param.grad when grad is None)
   - CORRECT: Only access .grad after loss.backward() in training loop, never in forward()
   - CORRECT: Use param.data or param directly in forward pass, not param.grad
   - If code tries to use gradients in forward(), remove that code - it will cause TypeError
12. Shape mismatch in normalization layers (CRITICAL - common with vision + transformer):
   - WRONG: Passing vision images [batch, channels, H, W] directly to transformer with LayerNorm expecting [batch, seq_len, d_model]
   - WRONG: Applying LayerNorm(d_model) to raw image tensor without flattening and projection
   - CORRECT: For vision + transformer models:
     * Flatten image: `x = x.view(batch_size, -1)` or `x = torch.flatten(x, 1)` (for MNIST: [128, 1, 28, 28] → [128, 784])
     * Project to d_model: Add `self.input_projection = nn.Linear(784, d_model)` in __init__, then `x = self.input_projection(x)`
     * Reshape for transformer: `x = x.view(batch_size, seq_len, d_model)` (e.g., [128, 784] → [128, 784, 512] if seq_len=784, or use patch embedding)
     * THEN apply LayerNorm: `x = self.norm(x)`
   - For MNIST (28x28=784 pixels): need projection from 784 to d_model, then reshape to [batch, seq_len, d_model] before LayerNorm
   - Error "Given normalized_shape=[512], expected input with shape [*, 512], but got input of size[128, 1, 28, 28]" means LayerNorm applied to wrong shape
   - Fix: Add input projection layer and reshape before normalization in forward() method
13. LoRA / Adaptation Layer Fixes (CRITICAL - IF CODE USES LORA/ADAPTERS):
   - **Fix Layer Traversal:**
     - WRONG: `for layer in model:` then `zip(lora_layers, model)` - Replace with unified layers list
     - CORRECT: Create `self.layers = nn.ModuleList()` that includes both LoRA-wrapped and original layers
     - CORRECT: `for layer in model: if isinstance(layer, nn.Linear): self.layers.append(LoRALinear(layer, rank)) else: self.layers.append(layer)`
     - CORRECT: `def forward(self, x): for layer in self.layers: x = layer(x); return x`
   - **Fix LoRA Math:**
     - WRONG: `return inputs + torch.matmul(inputs, self.A) @ self.B.T` - Replace with proper LoRA formula
     - WRONG: `torch.matmul(self.B, self.A.T)` - Fix to `self.B @ self.A` (formula is W' = W + B@A, NOT W + B@A^T)
     - WRONG: `self.A = nn.Parameter(torch.zeros(rank, layer.weight.shape[1]))` - Fix to `[rank, in_features]`
     - WRONG: `self.B = nn.Parameter(torch.zeros(layer.weight.shape[0], rank))` - Fix to `[out_features, rank]`
     - CORRECT: `self.A = nn.Parameter(torch.zeros(rank, layer.in_features))`  # [rank, in_features]
     - CORRECT: `self.B = nn.Parameter(torch.zeros(layer.out_features, rank))`  # [out_features, rank]
     - CORRECT: `return F.linear(x, self.layer.weight + self.B @ self.A, self.layer.bias)`  # W' = W + B@A (NOT B@A.T)
   - **Fix Model Wrapping:**
     - WRONG: `LoRAModel(ExampleModel())` where ExampleModel has custom forward() with Conv2d, flatten, etc.
     - WRONG: Using `model.children()` to iterate over custom model - breaks model structure
     - CORRECT: Use `nn.Sequential` for base model: `base_model = nn.Sequential(nn.Flatten(), nn.Linear(...), ...)`
     - CORRECT: Then wrap: `model = LoRAModel(base_model, rank=4)` - only works with Sequential or simple layer lists
   - **Remove Gradient Adjustment:**
     - WRONG: `adjust_gradients(A, B)` function with manual gradient math - Remove entirely
     - WRONG: `A.grad.data = dL_dA - torch.matmul(dL_dA, B) * B.T` - This is incorrect and causes shape errors
     - CORRECT: Remove all gradient adjustment code - LoRA gradients propagate automatically
     - CORRECT: Just use `loss.backward()` then `xm.optimizer_step(optimizer)` - no manual gradient adjustment
   - **Fix Loss Function:**
     - WRONG: `criterion = nn.CrossEntropyLoss().to(device)` - Remove .to(device)
     - CORRECT: `criterion = nn.CrossEntropyLoss()` (loss functions don't have parameters)
   - **Fix Training Loop:**
     - WRONG: Missing `model.train()` at start of each epoch - Add `model.train()` before training loop
     - WRONG: `int(labels.size(0))` on XLA tensors - Use `labels.size(0)` directly or `.sum().item()`
     - CORRECT: `model.train()` at start of each epoch, `model.eval()` only for evaluation
     - CORRECT: `total += labels.size(0)` and `correct += (predicted == labels).sum().item()`

14. Add any missing imports

CRITICAL: ONLY use VALID torch_xla.core.xla_model (xm) APIs:
   - xm.xla_device() - Get XLA device
   - xm.optimizer_step(optimizer) - XLA optimizer step (NO sync parameter!)
   - xm.mark_step() - Synchronize XLA computation
   - xm.rendezvous(tag) - Synchronization barrier (requires tag string)
   - xm.get_ordinal() - Get device ordinal (distributed)
   - xm.get_world_size() - Get world size (distributed)
   
   **CRITICAL: Regular PyTorch operations ARE supported in XLA:**
   - `torch.matmul()`, `torch.mm()`, `torch.add()`, `torch.mul()`, etc. - ALL work in XLA
   - `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, etc. - ALL standard PyTorch modules work in XLA
   - `nn.Module` - Use regular `nn.Module` for all classes, NOT `xm.XlaModule` (which doesn't exist)
   - XLA compatibility comes from using XLA device (`xm.xla_device()`) and XLA optimizer step, NOT from special APIs
   
   DO NOT use or suggest non-existent APIs like:
   - xm.optimizer.* (e.g., xm.optimizer.SGD) - THIS DOES NOT EXIST, use torch.optim.SGD instead
   - xm.XlaModule - THIS DOES NOT EXIST, use regular `nn.Module`
   - xm.dot() or xm.dot_general() - THESE DO NOT EXIST, use `torch.matmul()` or `torch.mm()`
   - xm.tensor() - THIS DOES NOT EXIST, use `torch.tensor()`
   - xm.scalar_tensor_to_python_scalar() - THIS DOES NOT EXIST, use `.item()` or `int()`
   - xm.xla_device_context() - THIS DOES NOT EXIST, use `device = xm.xla_device()` and `model.to(device)` instead
   - xm.mark_step_context() - THIS DOES NOT EXIST, just call `xm.mark_step()` directly
   - xm.send_cpu_data_to_device() - THIS DOES NOT EXIST, use `.to(device)` instead
   - xm.save_memory_state() - THIS DOES NOT EXIST
   - xm.optimizer_step(optimizer, sync=True) - sync parameter DOES NOT EXIST, just use xm.optimizer_step(optimizer)
   - Any other xm.* functions not listed above
   
   If you see `xm.optimizer.SGD` or `xm.optimizer.Adam` or similar, replace with `torch.optim.SGD` or `torch.optim.Adam` (regular PyTorch optimizers).
   Then use `xm.optimizer_step(optimizer)` instead of `optimizer.step()`.
   
   If you see `class MyModule(xm.XlaModule):`, replace with `class MyModule(nn.Module):` - XlaModule does NOT exist.
   
   If you see `xm.dot()` or `xm.dot_general()`, replace with `torch.matmul()` or `torch.mm()` - regular PyTorch operations work in XLA.
   
   If you see `xm.tensor()`, replace with `torch.tensor()`.
   
   If you see `xm.scalar_tensor_to_python_scalar()`, replace with `.item()` for scalar tensors or `int()` for size conversions.
   
   If you see `with xm.xla_device_context():` or `with xm.mark_step_context():`, remove the context manager and use the direct API calls instead.
   
   If you see `xm.send_cpu_data_to_device(...)`, replace with `.to(device)` (e.g., `criterion.to(device)`).
   
   If you see `xm.optimizer_step(optimizer, sync=True)`, remove the sync parameter - just use `xm.optimizer_step(optimizer)`.
   
   If you see code using non-existent xm.* APIs, remove them - they will cause AttributeError.

CRITICAL FIXING REQUIREMENTS - YOU MUST ACTUALLY FIX THE CODE:

1. **MUST IMPLEMENT MISSING FUNCTIONS**: If an issue says a function is "not implemented" or has `pass`, you MUST write the complete implementation. Do NOT leave `pass`, `# TODO`, or empty function bodies.

2. **MUST FIX ELLIPSIS PLACEHOLDERS (CRITICAL)**: If you see `variable = ...` or `variable = ... # comment`, you MUST replace it with actual initialization code. The ellipsis (`...`) is NOT a valid value - it's a placeholder that will cause AttributeError at runtime. Examples:
   - WRONG: `base_model = ... # Initialize your base model here`
   - CORRECT: `base_model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))`
   - WRONG: `model = ...`
   - CORRECT: `model = MyModel()` or `model = nn.Linear(10, 5)` or appropriate initialization
   - Check ALL variable assignments - if any use `...`, replace with proper initialization

3. **MUST CORRECT INCORRECT CODE**: If an issue says something is "incorrect" or "wrong", you MUST replace it with the correct implementation. Do NOT keep the incorrect code.

4. **MUST FIX ALL ISSUES**: Address EVERY issue listed above. Do NOT skip any issues. The fixed code should NOT have any of the problems identified.

5. **MUST RETURN COMPLETE CODE**: Return the ENTIRE fixed Python file, not just the parts you changed. Include ALL imports, ALL classes, ALL functions, and ALL execution code.

6. **VERIFY YOUR FIXES**: Before returning, check that:
   - All functions are fully implemented (no `pass` statements)
   - All ellipsis (`...`) placeholders are replaced with actual code
   - All incorrect implementations are corrected
   - All missing functionality is added
   - The code will actually run without the errors identified
   - NO variable is assigned to `...` - this will cause AttributeError at runtime

IMPORTANT: The next iteration will review YOUR fixed code. If you don't actually fix the issues, the same problems will be found again. You MUST implement real fixes, not placeholders.

{iterative_note}

⚠️ CRITICAL: The code you return MUST be DIFFERENT from the code above. If you return identical code, the fix will be considered failed. You MUST make actual changes to address the issues listed. Do NOT return the same code - make real modifications.

Return ONLY the complete fixed Python code in a code block. Do not include explanations outside the code block. The code block should contain the ENTIRE fixed file."""
        
        # Format the prompt with actual values (escaping braces in code)
        iteration_minus_one = max(0, iteration - 1)
        # Format fix_history first if it exists (it contains {iteration} placeholder)
        formatted_fix_history = fix_history.format(iteration=iteration) if fix_history else ""
        # Add iterative note if this is not the first iteration
        iterative_note = ""
        if iteration > 1:
            iterative_note = "\n⚠️ CRITICAL: Since this is iteration {} and previous fixes were attempted, you MUST:\n".format(iteration)
            iterative_note += "1. Review what was tried in previous iterations (see fix history above)\n"
            iterative_note += "2. Understand WHY those fixes didn't work (they were incomplete or incorrect)\n"
            iterative_note += "3. Provide DIFFERENT and BETTER fixes that actually solve the root cause\n"
            iterative_note += "4. Do NOT repeat the same ineffective fixes - try a different approach\n"
        fix_prompt = fix_prompt_template.format(
            fix_history=formatted_fix_history,
            trainium_errors=trainium_errors_ref,
            iteration=iteration,
            iteration_minus_one=iteration_minus_one,
            issues_text=issues_text,
            fixes_text=fixes_text,
            code=code.replace('{', '{{').replace('}', '}}'),
            dataset_name=dataset_name or 'unknown',
            iterative_note=iterative_note
        )

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
            # IMPORTANT: Use findall to get ALL code blocks, then use the LAST one (which should be the fixed code)
            # The AI might include the original code for context, so we want the final code block
            code_patterns = [
                r'```python\n(.*?)```',  # Python code block (non-greedy to match each block separately)
                r'```\n(.*?)```',  # Generic code block
            ]
            
            extracted_code = None
            all_code_blocks = []
            
            # Find all code blocks using each pattern
            for pattern in code_patterns:
                matches = re.findall(pattern, fixed_text, re.DOTALL)
                for match in matches:
                    candidate = match.strip()
                    # Validate it's actually code (has imports or def/class)
                    if ('import' in candidate or 'def ' in candidate or 'class ' in candidate) and len(candidate) > 100:
                        all_code_blocks.append(candidate)
            
            # If we found multiple code blocks, use the LAST one (should be the fixed code)
            # Also prefer longer code blocks (more likely to be complete)
            if all_code_blocks:
                if len(all_code_blocks) > 1:
                    logger.info(f"Found {len(all_code_blocks)} code blocks in response, using the last one (should be fixed code)")
                    # Use the last code block, but prefer longer ones if they're significantly different
                    extracted_code = all_code_blocks[-1]
                    # Check if any block is significantly longer (likely the complete fixed code)
                    longest_block = max(all_code_blocks, key=len)
                    if len(longest_block) > len(extracted_code) * 1.2:  # 20% longer
                        logger.info(f"Using longest code block instead (length: {len(longest_block)} vs {len(extracted_code)})")
                        extracted_code = longest_block
                else:
                    extracted_code = all_code_blocks[0]
                logger.info(f"✅ Extracted code block ({len(extracted_code)} chars)")
            
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
                # Validate the extracted code is different from original
                # Use normalized comparison (strip whitespace differences)
                original_normalized = code.strip()
                extracted_normalized = extracted_code.strip()
                
                if extracted_normalized != original_normalized:
                    # Additional check: ensure it's not just whitespace differences
                    # Compare without whitespace to catch cases where only formatting changed
                    original_no_ws = ''.join(original_normalized.split())
                    extracted_no_ws = ''.join(extracted_normalized.split())
                    
                    if extracted_no_ws != original_no_ws:
                        logger.info(f"✅ Successfully extracted fixed code ({len(extracted_code)} chars, original: {len(code)} chars)")
                        return extracted_code
                    else:
                        logger.warning("Extracted code differs only in whitespace - no substantive changes made")
                        logger.debug(f"Original length: {len(code)}, Extracted length: {len(extracted_code)}")
                else:
                    logger.warning("Extracted code is identical to original - no changes made")
                    logger.debug(f"Both codes are {len(extracted_code)} characters")
                    # Log a snippet to help debug what the AI returned
                    if len(fixed_text) > 1000:
                        logger.debug(f"AI response preview (first 500 chars): {fixed_text[:500]}...")
                        logger.debug(f"AI response preview (last 500 chars): ...{fixed_text[-500:]}")
            else:
                logger.error("Could not extract fixed code from AI response")
                logger.error(f"Response preview (first 500 chars): {fixed_text[:500]}...")
                logger.error(f"Response preview (last 500 chars): ...{fixed_text[-500:]}")
            
            # If we can't extract code, return None (fix failed)
            return None
            
        except Exception as e:
            logger.error(f"Error in AI code fix: {e}")
            return None

