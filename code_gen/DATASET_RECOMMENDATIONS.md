# Dataset Recommendation System

## Overview

The dataset recommendation system automatically analyzes research papers to determine the most appropriate datasets for training models described in the papers. This ensures that generated PyTorch code uses relevant, open-source datasets that match the paper's domain and requirements.

## Features

1. **Automatic Dataset Detection**
   - Extracts explicitly mentioned datasets from paper text (e.g., "CIFAR-10", "IMDB")
   - Uses regex patterns to find dataset mentions in abstracts and full paper content

2. **Domain Inference**
   - Analyzes paper content to infer domain (Computer Vision, NLP, General)
   - Uses keyword matching to determine task type

3. **LLM-Based Recommendations**
   - Uses AWS Bedrock Claude to intelligently recommend datasets based on paper analysis
   - Considers paper domain, task type, and requirements
   - Provides reasoning for recommendations

4. **Prioritized Selection**
   - Priority 1: Explicitly mentioned datasets (if available in our system)
   - Priority 2: LLM recommendations
   - Priority 3: Domain-based recommendations
   - Fallback: Synthetic dataset

## Available Datasets

Currently supported datasets:
- **cifar10**: 60K 32x32 color images, 10 classes (computer vision)
- **cifar100**: 60K 32x32 color images, 100 classes (fine-grained classification)
- **mnist**: 70K 28x28 grayscale digits (simple baselines)
- **fashion_mnist**: 70K 28x28 grayscale fashion items
- **imdb**: 50K movie reviews (NLP, sentiment analysis)
- **wikitext2**: Language modeling dataset (transformers, LLMs)
- **synthetic**: Generated synthetic data (quick testing)

## How It Works

### 1. Paper Analysis

When generating code for a paper:

```python
# In pytorch_generator.py
dataset_recommendations = self.dataset_recommender.recommend_datasets(
    paper, paper_content, use_llm=True
)
```

### 2. Recommendation Process

The system:
1. Extracts dataset mentions from paper text
2. Infers domain from keywords (vision, NLP, etc.)
3. Queries LLM for intelligent recommendations (optional)
4. Prioritizes and combines recommendations
5. Selects primary dataset

### 3. Code Generation

Recommended datasets are included in the Bedrock prompt:

```
DATASET RECOMMENDATIONS (IMPORTANT - USE THESE):
Based on analysis of this paper, the following datasets are recommended:

PRIMARY DATASET: cifar10
RECOMMENDED DATASETS: cifar10, mnist
INFERRED DOMAIN: vision
EXPLICITLY MENTIONED IN PAPER: CIFAR-10
REASONING: Paper focuses on image classification tasks...
```

### 4. Generated Code

The generated code uses the recommended dataset:

```python
from dataset_loader import load_dataset

# Use the recommended primary dataset: cifar10
train_loader, test_loader = load_dataset('cifar10', batch_size=128)
print(f"Using recommended dataset: cifar10")
```

## Integration

### In PyTorch Code Generator

Dataset recommendations are automatically integrated into the code generation pipeline:

```python
# pytorch_generator.py
def generate_code_for_paper(self, paper_id: str, include_full_content: bool = False):
    # ... get paper ...
    
    # Get dataset recommendations
    dataset_recommendations = self.dataset_recommender.recommend_datasets(
        paper, paper_content, use_llm=True
    )
    
    # Generate code with recommendations
    result = self.bedrock_client.generate_pytorch_code(
        paper_summary, 
        paper_content,
        dataset_recommendations=dataset_recommendations
    )
    
    # Include recommendations in result
    result["dataset_recommendations"] = dataset_recommendations
    result["recommended_dataset"] = dataset_recommendations.get("primary_dataset")
```

### Metadata Storage

Dataset recommendations are stored in:
- S3 metadata JSON files
- OpenSearch document fields:
  - `recommended_dataset`: Primary dataset name
  - `dataset_recommendations`: Full recommendation dictionary

## Example Output

```json
{
  "paper_id": "abc123",
  "recommended_dataset": "cifar10",
  "dataset_recommendations": {
    "recommended_datasets": ["cifar10", "mnist"],
    "primary_dataset": "cifar10",
    "domain": "vision",
    "explicitly_mentioned": ["cifar10"],
    "llm_reasoning": "Paper focuses on image classification with CNNs...",
    "confidence": "high"
  }
}
```

## Adding New Datasets

To add support for new datasets:

1. **Update `dataset_recommender.py`**:
   - Add dataset to `DATASET_NAME_MAP`
   - Add pattern to `DATASET_PATTERNS`
   - Add to `DATASET_KNOWLEDGE_BASE` by domain
   - Add to `AVAILABLE_DATASETS`

2. **Update `dataset_loader.py`**:
   - Add loader function
   - Register in `_build_loader_registry()`

3. **Update Bedrock prompt** (if needed):
   - Add dataset description to available datasets list

## Configuration

### Enable/Disable LLM Recommendations

```python
# Use LLM for recommendations (default: True)
dataset_recommendations = recommender.recommend_datasets(
    paper, paper_content, use_llm=True
)

# Use only rule-based recommendations
dataset_recommendations = recommender.recommend_datasets(
    paper, paper_content, use_llm=False
)
```

## Benefits

1. **Automatic Selection**: No manual dataset selection required
2. **Intelligent Matching**: LLM ensures appropriate dataset choice
3. **Domain Awareness**: Understands paper context and requirements
4. **Explicit Detection**: Finds datasets mentioned in papers
5. **Fallback Support**: Always provides a valid dataset option

## Future Enhancements

- Support for more datasets (ImageNet, COCO, SQuAD, etc.)
- Multi-dataset recommendations for papers using multiple datasets
- Dataset-specific preprocessing recommendations
- Integration with HuggingFace datasets
- Automatic dataset download and caching for new datasets

