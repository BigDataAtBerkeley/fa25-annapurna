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
        self.max_iterations = 7  # Run up to 3 iterations to catch and fix issues
        self.fix_history = []  # Track fixes applied
    
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
        review_start_time = time.time()
        
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
            
            # Analyze code for issues (using current_code which may have been fixed in previous iteration)
            logger.debug(f"Iteration {iteration}: Analyzing code ({len(current_code)} chars)")
            issues = self._analyze_code_with_ai(current_code, dataset_name, fixes_applied)
            
            if not issues or issues.get('no_issues', False):
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
                    'fixes': fixes_needed  # Use fixes_needed, not fixes
                })
                # Log code changes for debugging
                logger.info(f"Code length changed: {len(current_code)} -> {len(fixed_code)} chars")
                # Check if the specific issues mentioned are actually fixed
                for issue in issues_found[:2]:  # Check first 2 issues
                    if '@ operator' in issue.lower() and '@' in fixed_code:
                        logger.warning(f"⚠️ Issue '{issue[:50]}...' may not be fixed - '@' still present in code")
                # CRITICAL: Update current_code so next iteration uses the fixed code
                current_code = fixed_code
                logger.info(f"✅ Applied fixes in iteration {iteration} - code updated (next iteration will use this fixed code)")
            elif fixed_code == current_code:
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
        
        review_time = time.time() - review_start_time
        logger.info(f"Code review completed in {review_time:.2f}s ({iteration} iterations)")
        
        return {
            'code': current_code,
            'fixes_applied': fixes_applied,
            'iterations': iteration,
            'review_time': review_time,
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
        
        # Use .format() instead of f-string to avoid issues with curly braces in code
        # Escape JSON braces in the template, then format with actual values
        trainium_errors_ref = self._get_trainium_error_reference()
        analysis_prompt_template = """You are an expert code reviewer for PyTorch code that will run on AWS Trainium using the Neuron SDK (XLA devices).

Your task: Review this PyTorch code and identify ANY issues that would prevent it from running correctly on Trainium with the Neuron SDK. Look for ALL types of errors, not just common ones.

CRITICAL: This code MUST use the Neuron SDK (torch_xla) for Trainium execution. Ensure all XLA operations are correct.

```python
{code}
```

Dataset being used: {dataset_name}

{fix_history}

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
            for fix in previous_fixes:
                prev_iter = fix.get('iteration', '?')
                prev_issues = fix.get('issues_found', [])
                prev_fixes = fix.get('fixes', [])
                prev_fixes_str = ', '.join(prev_fixes[:2]) if prev_fixes else 'none'
                fix_history += "  - Iteration {}: Found {} issues, attempted fixes: {}...\n".format(
                    prev_iter, len(prev_issues), prev_fixes_str
                )
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
13. Add any missing imports

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

Return ONLY the complete fixed Python code in a code block. Do not include explanations outside the code block."""
        
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
                # Validate the extracted code is different from original
                if extracted_code != code:
                    logger.info(f"✅ Successfully extracted fixed code ({len(extracted_code)} chars)")
                    return extracted_code
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

