# Code Generation System - Description Guide for Midpoint Deliverable

This guide provides a framework for describing how code generation works in your midpoint deliverable, including diagrams, text descriptions, and results to include.

**Important Note**: This deliverable uses `pipeline_for_delivery.py` which runs as a direct Python script. It does NOT use Lambda functions or SQS queues. Results are saved locally to the `results/` directory. The `lambda_handler.py` file exists in the codebase but is not used in this deliverable.

## 1. High-Level Architecture Overview

### System Architecture Diagram (Text Description)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODE GENERATION PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  OpenSearch  │  ← Research papers indexed with metadata
    │   (Papers)    │     and full content in S3
    └──────┬───────┘
           │
           │ 1. Paper Retrieval
           ▼
    ┌─────────────────────┐
    │  OpenSearchClient    │  ← Retrieves paper metadata + content
    │  - get_paper_by_id()│
    │  - get_paper_content()│
    └──────┬──────────────┘
           │
           │ 2. Dataset Recommendation
           ▼
    ┌─────────────────────┐
    │ DatasetRecommender  │  ← Analyzes paper to recommend datasets
    │  - Pattern matching │     (CIFAR-10, IMDB, MNIST, etc.)
    │  - LLM analysis     │
    └──────┬──────────────┘
           │
           │ 3. Code Generation
           ▼
    ┌─────────────────────┐
    │   BedrockClient     │  ← AWS Bedrock (Claude 3 Sonnet)
    │  - generate_pytorch_code()│  Generates PyTorch code with
    │                      │     Neuron SDK requirements
    └──────┬──────────────┘
           │
           │ 4. Code Review & Fix
           ▼
    ┌─────────────────────┐
    │  CodeReviewAgent    │  ← Static analysis + AI review
    │  - Static checks    │     Fixes Neuron SDK issues
    │  - AI analysis      │     Handles XLA compatibility
    │  - Iterative fixes  │
    └──────┬──────────────┘
           │
           │ 5. Output
           ▼
    ┌─────────────────────┐
    │  Generated Code     │  → Ready for Trainium execution
    │  + Metadata         │
    └─────────────────────┘
```

### Key Components

1. **OpenSearchClient**: Retrieves paper metadata and full content from S3
2. **DatasetRecommender**: Intelligently recommends datasets (CIFAR-10, IMDB, etc.) using pattern matching + LLM
3. **BedrockClient**: Generates PyTorch code using Claude 3 Sonnet with Neuron SDK requirements
4. **CodeReviewAgent**: Reviews and fixes code for Trainium compatibility (XLA operations, imports, etc.)

---

## 2. Detailed Flow Description

### Pipeline Initiation

The pipeline is initiated via command line by running `pipeline_for_delivery.py`:

```bash
python pipeline_for_delivery.py [options]
```

**Entry Point**: `main()` function in `pipeline_for_delivery.py` (line 638)

**Paper Selection Options**:
1. **Specific Paper**: `--paper-id <paper_id>` - Process one paper by ID
2. **Recent Papers**: `--recent-days 30 --max-papers 5` - Get papers from last N days (default)
3. **Random Papers**: `--random --max-papers 5` - Get random papers from OpenSearch
4. **Custom Query**: `--query '{"match": {"title": "ResNet"}}' --max-papers 3` - Custom OpenSearch query

**Flow**:
1. `main()` initializes `PyTorchCodeGenerator()`
2. Uses `generator.opensearch_client` to get paper IDs:
   - `get_recent_papers()` - Default: recent papers
   - `get_random_papers()` - Random selection
   - `search_papers()` - Custom query
   - Direct paper ID if provided
3. For each paper ID, calls `process_paper(paper_id, generator)`
4. `process_paper()` orchestrates the full pipeline

**Visual Flow**:
```
Command Line
    │
    ├─ python pipeline_for_delivery.py --paper-id <id>
    ├─ python pipeline_for_delivery.py --recent-days 30 --max-papers 5
    ├─ python pipeline_for_delivery.py --random --max-papers 5
    └─ python pipeline_for_delivery.py --query '{"match": {...}}'
    │
    ▼
main() [pipeline_for_delivery.py:638]
    │
    ├─ Initialize PyTorchCodeGenerator()
    │
    ├─ Get paper IDs from OpenSearch:
    │   ├─ generator.opensearch_client.get_recent_papers()
    │   ├─ generator.opensearch_client.get_random_papers()
    │   ├─ generator.opensearch_client.search_papers()
    │   └─ Direct paper ID (if --paper-id provided)
    │
    └─ For each paper_id:
        │
        ▼
    process_paper(paper_id, generator) [pipeline_for_delivery.py:345]
        │
        ├─ Step 1: Paper Retrieval
        │   └─ generator.opensearch_client.get_paper_by_id()
        │   └─ generator.opensearch_client.get_paper_content()
        │
        ├─ Step 2-4: Code Generation & Review
        │   └─ generator.generate_code_for_paper()
        │       ├─ Dataset Recommendation
        │       ├─ Bedrock Code Generation
        │       └─ Code Review & Fix
        │
        └─ Step 5: Save Results
            └─ results/{paper_id}/
```

### Step-by-Step Process

#### **Step 1: Paper Retrieval**
- **Location**: `process_paper()` function (line 345) → `generator.opensearch_client`
- **Input**: Paper ID (from command line args or OpenSearch query)
- **Process**: 
  - `generator.opensearch_client.get_paper_by_id(paper_id)` retrieves paper metadata from OpenSearch (title, authors, abstract)
  - `generator.opensearch_client.get_paper_content(paper)` downloads full paper text from S3 (up to 150K chars)
  - `generator.opensearch_client.get_paper_summary(paper)` formats paper summary
- **Output**: Paper summary + full content
- **Saved to**: `results/{paper_id}/paper-retrieval/{paper_id}_{timestamp}.json`

#### **Step 2: Dataset Recommendation**
- **Location**: Inside `generator.generate_code_for_paper()` (called from `process_paper()` line 399)
- **Input**: Paper metadata + content
- **Process**:
  - **Pattern Matching**: Extracts explicitly mentioned datasets (CIFAR-10, IMDB, etc.) using regex patterns
  - **LLM Analysis**: Uses Bedrock to intelligently recommend datasets based on:
    - Paper domain (vision, NLP, etc.)
    - Task type (classification, language modeling, etc.)
    - Available datasets in system
  - **Prioritization**: Combines explicit mentions + LLM recommendations, filters to available datasets
- **Output**: Primary dataset + list of recommended datasets + reasoning

#### **Step 3: Code Generation**
- **Location**: `generator.generate_code_for_paper()` (called from `process_paper()` line 399) → `bedrock_client.generate_pytorch_code()`
- **Input**: Paper summary, full content, dataset recommendations
- **Process**:
  - **Prompt Engineering**: Creates comprehensive prompt with:
    - Paper information (title, authors, abstract, full content)
    - Dataset recommendations
    - **Critical Neuron SDK requirements**:
      - Must use `torch_xla.core.xla_model as xm`
      - Must use `xm.xla_device()` for Trainium device
      - Must use `xm.optimizer_step()` instead of `optimizer.step()`
      - Must call `xm.mark_step()` after backward pass
      - All tensors moved to device before operations
    - Common pitfalls and fixes
    - Training loop template
  - **Bedrock API Call**: 
    - Model: Claude 3 Sonnet (`anthropic.claude-3-sonnet-20240229-v1:0`)
    - Max tokens: 8,192 (output limit)
    - Temperature: 0.3 (lower for deterministic code)
    - Retry logic: 8 attempts with exponential backoff for throttling
  - **Code Extraction**: Parses code blocks from Claude's response
- **Output**: Generated PyTorch code + explanation

#### **Step 4: Code Review & Fix**
- **Location**: Inside `generator.generate_code_for_paper()` (line 98) → `code_review_agent.review_and_fix_code()`
- **Input**: Generated code + dataset name
- **Process**:
  - **Static Analysis** (Quick fixes):
    - Checks for incomplete/truncated code (unclosed brackets, incomplete statements)
    - Validates XLA operations (`xm.optimizer_step`, `xm.mark_step` present)
    - Checks for missing imports (math, random, collections)
    - Validates dataset-specific requirements (IMDB tokenization, etc.)
    - Fixes invalid imports (`transformers_xla` → `transformers`)
    - Fixes `xm.rendezvous()` without tag argument
  - **AI Analysis**:
    - Uses Bedrock to comprehensively review code for:
      - Neuron SDK/XLA compatibility
      - Data handling issues
      - Model output handling
      - Type errors
      - Logic errors
      - XLA-specific gotchas (tensor.size(0) returns tensor, not int)
    - Returns JSON with issues list + fix instructions
  - **AI Fix**:
    - Uses Bedrock to apply fixes based on identified issues
    - Returns complete fixed code
  - **Iteration**: Currently 1 iteration (can be increased)
- **Output**: Reviewed and fixed code + fix history

#### **Step 5: Result Storage**
- **Location**: `process_paper()` function saves results at each step
- **Local File System**: Saves to `results/{paper_id}/` with subdirectories:
  - `code-generation/` - Generated code + metadata
  - `code-review/` - Reviewed code + fix history
  - `trn-execution/` - Execution results (if Trainium execution enabled)
  - `metrics/` - Training metrics (if execution successful)
  - `profiler/` - Profiler metadata (if profiler enabled)
- **File Format**:
  - Code files: `{paper_id}_{timestamp}.py`
  - Metadata: `{paper_id}_{timestamp}.json`
- **Note**: The midpoint deliverable uses direct file storage, not Lambda/SQS integration

---

## 3. Key Technical Details to Highlight

### Neuron SDK Integration
- **Why**: Code must run on AWS Trainium hardware, which requires Neuron SDK (XLA)
- **Key Requirements**:
  - `torch_xla.core.xla_model as xm` - XLA interface for Trainium
  - `xm.xla_device()` - Gets Trainium accelerator device (NOT CUDA)
  - `xm.optimizer_step(optimizer)` - Required XLA optimizer step
  - `xm.mark_step()` - Synchronizes XLA computation after backward pass
  - All tensors moved to device BEFORE operations
  - Loss functions NOT moved to device
  - Tensor size operations: `tensor.size(0)` returns tensor in XLA, must use `int()` for arithmetic

### Prompt Engineering
- **Comprehensive Prompt**: ~400 lines of detailed instructions
- **Includes**:
  - Paper information
  - Dataset recommendations
  - Neuron SDK requirements (critical section)
  - Available packages list
  - Common pitfalls to avoid
  - Training loop template
  - Output format requirements
- **Token Limit Handling**: 8,192 token output limit, detects truncation

### Error Handling & Robustness
- **Bedrock Throttling**: 8 retries with exponential backoff (5s → 120s max)
- **Timeout Handling**: 150s Bedrock timeout, 180s per-paper timeout
- **Truncation Detection**: Detects incomplete code from max_tokens limit
- **Code Review**: Catches issues before execution

### Dataset Intelligence
- **Pattern Matching**: Extracts 20+ dataset name patterns from paper text
- **LLM Reasoning**: Uses Bedrock to intelligently recommend based on domain/task
- **Available Datasets**: cifar10, cifar100, mnist, fashion_mnist, imdb, wikitext2, synthetic
- **Domain-Aware**: Recommends NLP datasets (imdb, wikitext2) for NLP papers, vision datasets for vision papers

---

## 4. Results & Metrics to Include

### Quantitative Results

#### Code Generation Success Rate
- **Example**: "Generated code for X papers, Y% success rate"
- **Metrics**:
  - Total papers processed
  - Successful generations
  - Failed generations (with reasons)
  - Average generation time per paper

#### Code Quality Metrics
- **Code Review Fixes**:
  - Average fixes per paper
  - Most common issues found
  - Fix success rate
- **Execution Success**:
  - % of generated code that executes successfully on Trainium
  - Common execution errors

#### Performance Metrics
- **Generation Time**: Average time per paper (include Bedrock API time)
- **Review Time**: Average code review time
- **Total Pipeline Time**: End-to-end time per paper

### Qualitative Results

#### Example Generated Code
- Show a complete example of generated code
- Highlight Neuron SDK integration
- Show before/after code review fixes

#### Dataset Recommendation Examples
- Show examples of intelligent dataset recommendations
- Include LLM reasoning for recommendations

#### Code Review Examples
- Show examples of issues found and fixes applied
- Highlight XLA-specific fixes

### Visualizations to Include

1. **Pipeline Flow Diagram** (see Section 1)
2. **Component Interaction Diagram**: Show how components interact
3. **Code Generation Timeline**: Show time spent in each step
4. **Success Rate Chart**: Bar chart of success/failure rates
5. **Issue Distribution Chart**: Most common code review issues
6. **Dataset Recommendation Distribution**: Which datasets are recommended most

---

## 5. Sample Text Descriptions

### Executive Summary (1-2 paragraphs)

"Our code generation system automatically converts research papers into executable PyTorch code optimized for AWS Trainium hardware. The system uses AWS Bedrock (Claude 3 Sonnet) to generate code from paper content, intelligently recommends appropriate datasets, and performs automated code review to ensure Trainium compatibility. The generated code uses the Neuron SDK (XLA) for Trainium execution, with automatic fixes for common compatibility issues."

### Detailed Description (2-3 pages)

**Introduction**
The code generation system is a multi-stage pipeline that transforms research papers into production-ready PyTorch code. The system addresses the challenge of implementing paper algorithms correctly, especially for specialized hardware like AWS Trainium.

**Architecture**
The system consists of four main components:
1. **Paper Retrieval**: Retrieves paper metadata from OpenSearch and full content from S3
2. **Dataset Recommendation**: Intelligently recommends datasets using pattern matching and LLM analysis
3. **Code Generation**: Uses AWS Bedrock to generate PyTorch code with Neuron SDK requirements
4. **Code Review**: Performs static analysis and AI-powered review to fix compatibility issues

The pipeline runs as a direct Python script (`pipeline_for_delivery.py`), saving results locally to the `results/` directory. No Lambda functions or SQS queues are used in this deliverable.

**Code Generation Process**
[Describe the detailed flow from Section 2]

**Neuron SDK Integration**
[Describe Neuron SDK requirements and why they're critical]

**Results**
[Include quantitative and qualitative results from Section 4]

---

## 6. Diagrams to Create

### 1. System Architecture Diagram
- Show all components and their relationships
- Include data flow arrows
- Label each component

### 2. Code Generation Flow Diagram
- Show step-by-step process
- Include decision points (e.g., "Code review finds issues?")
- Show outputs at each step

### 3. Component Interaction Sequence Diagram
- Show how components interact over time
- Include API calls (Bedrock, OpenSearch for paper retrieval, S3 for paper content)

### 4. Prompt Structure Diagram
- Show how the prompt is constructed
- Highlight critical sections (Neuron SDK requirements)

### 5. Code Review Process Diagram
- Show static analysis → AI analysis → AI fix flow
- Include iteration loop

---

## 7. Code Examples to Include

### Example 1: Generated Code Snippet
```python
# Show a complete example with:
# - Imports (including torch_xla)
# - Model definition
# - Dataset loading
# - Training loop with XLA operations
# - Evaluation
```

### Example 2: Before/After Code Review
```python
# BEFORE (has issues):
# - Missing xm.mark_step()
# - tensor.size(0) used without int()
# - Missing import math

# AFTER (fixed):
# - All XLA operations correct
# - All imports present
# - Type conversions correct
```

### Example 3: Prompt Excerpt
```python
# Show a snippet of the prompt highlighting:
# - Neuron SDK requirements section
# - Training loop template
# - Common pitfalls section
```

---

## 8. Key Statistics to Report

- **Papers Processed**: Total number
- **Success Rate**: % of successful code generations
- **Average Generation Time**: Per paper
- **Code Review Fixes**: Average fixes per paper
- **Most Common Issues**: Top 5 issues found
- **Dataset Recommendations**: Distribution of recommended datasets
- **Execution Success Rate**: % of code that executes successfully on Trainium

---

## 9. Challenges & Solutions

### Challenge 1: Neuron SDK Compatibility
- **Problem**: Generated code often uses CUDA or standard PyTorch operations
- **Solution**: Comprehensive prompt engineering + automated code review

### Challenge 2: Code Truncation
- **Problem**: 8,192 token output limit can truncate code
- **Solution**: Truncation detection + warnings + code review catches incomplete code

### Challenge 3: Dataset Selection
- **Problem**: Papers may not explicitly mention datasets
- **Solution**: LLM-based intelligent recommendation based on domain/task

### Challenge 4: Bedrock Rate Limits
- **Problem**: Throttling errors during high-volume generation
- **Solution**: Exponential backoff retry logic (8 attempts, up to 120s delay)

---

## 10. Future Improvements

- Increase code review iterations (currently 1, could be 2-3)
- Add execution feedback loop (use execution errors to improve generation)
- Support for more datasets
- Fine-tuned prompts for specific paper types
- Multi-model ensemble (generate with multiple models, select best)

---

## Appendix: File Structure

```
code-gen-for-deliv/
├── pytorch_generator.py        # Main orchestrator (used by pipeline)
├── bedrock_client.py           # Bedrock API client + prompt engineering
├── code_review_agent.py        # Code review + fixes
├── dataset_recommender.py      # Dataset recommendation logic
├── opensearch_client.py        # Paper retrieval (from OpenSearch + S3)
├── lambda_handler.py           # Lambda entry point (NOT used in deliverable)
└── utils.py                   # Utility functions

pipeline_for_delivery.py        # Main pipeline script (direct execution)
```

**Note**: The midpoint deliverable uses `pipeline_for_delivery.py` which directly calls `PyTorchCodeGenerator`. The `lambda_handler.py` exists for Lambda deployment but is not used in this deliverable.

---

## Quick Reference: Key Numbers

- **Bedrock Model**: Claude 3 Sonnet (`anthropic.claude-3-sonnet-20240229-v1:0`)
- **Max Output Tokens**: 8,192
- **Temperature**: 0.3
- **Bedrock Timeout**: 150 seconds
- **Per-Paper Timeout**: 180 seconds
- **Retry Attempts**: 8 (with exponential backoff)
- **Code Review Iterations**: 1 (configurable)
- **Available Datasets**: 7 (cifar10, cifar100, mnist, fashion_mnist, imdb, wikitext2, synthetic)
- **Prompt Length**: ~400 lines
- **Static Checks**: 10+ issue types detected

