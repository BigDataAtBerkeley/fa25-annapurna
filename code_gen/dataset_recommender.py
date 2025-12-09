"""
Dataset Recommender for Research Papers

This module analyzes research papers to recommend appropriate datasets
for training models described in the papers.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Set
import boto3
from botocore.exceptions import ClientError

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


DATASET_KNOWLEDGE_BASE = {
    # Computer Vision - Image Classification
    "vision": ["cifar10", "cifar100", "mnist", "fashion_mnist", "imagenet"],
    "image_classification": ["cifar10", "cifar100", "mnist", "fashion_mnist"],
    "object_detection": ["coco", "pascal_voc"],
    "semantic_segmentation": ["cityscapes", "pascal_voc"],
    
    # NLP
    "nlp": ["imdb", "wikitext2", "mnist"],
    "text_classification": ["imdb", "mnist"],  
    "sentiment_analysis": ["imdb", "mnist"],  
    "language_modeling": ["wikitext2", "mnist"], 
    "question_answering": ["squad", "squad2"],
    "machine_translation": ["wmt", "iwslt"],
    
    # General / Testing
    "general": ["mnist"],
    "testing": ["mnist"]
}

# Dataset name patterns to extract from paper text
DATASET_PATTERNS = [
    r'\b(CIFAR-10|CIFAR10|CIFAR-100|CIFAR100)\b',
    r'\b(MNIST|Fashion-MNIST|FashionMNIST)\b',
    r'\b(ImageNet|ILSVRC)\b',
    r'\b(COCO|Common Objects in Context)\b',
    r'\b(PASCAL VOC|VOC)\b',
    r'\b(IMDB|IMDb)\b',
    r'\b(SQuAD|Stanford Question Answering Dataset)\b',
    r'\b(AG News|AGNews)\b',
    r'\b(WikiText|WikiText-2|WikiText2)\b',
    r'\b(Penn Treebank|PTB)\b',
    r'\b(BookCorpus|bookcorpus)\b',
    r'\b(GLU|GLUE)\b',
    r'\b(SST|Stanford Sentiment Treebank)\b',
    r'\b(Yelp)\b',
    r'\b(WMT|Workshop on Machine Translation)\b',
    r'\b(IWSLT|International Workshop on Spoken Language Translation)\b',
    r'\b(Cityscapes)\b',
]

# Normalized dataset name mapping
DATASET_NAME_MAP = {
    "cifar-10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar100": "cifar100",
    "mnist": "mnist",
    "fashion-mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
    "imagenet": "imagenet",
    "coco": "coco",
    "pascal voc": "pascal_voc",
    "voc": "pascal_voc",
    "imdb": "imdb",
    "squad": "squad",
    "ag news": "ag_news",
    "agnews": "ag_news",
    "wikitext": "wikitext2",
    "wikitext-2": "wikitext2",
    "wikitext2": "wikitext2",
    "penn treebank": "ptb",
    "ptb": "ptb",
    "bookcorpus": "bookcorpus",
    "glue": "glue",
    "sst": "sst2",
    "sst2": "sst2",
    "yelp": "yelp",
    "wmt": "wmt",
    "iwslt": "iwslt",
    "cityscapes": "cityscapes",
    "synthetic": "synthetic"
}

# Available datasets in our system (from dataset_loader)
# Vision datasets use .pt files, NLP datasets use HuggingFace Arrow format
AVAILABLE_DATASETS = {
    "cifar10", "cifar100", "mnist", "fashion_mnist", "imdb", "wikitext2", "synthetic"
}


class DatasetRecommender:
    """Recommends datasets based on paper content analysis."""
    
    def __init__(self, bedrock_client=None):
        """
        Initialize the dataset recommender.
        
        Args:
            bedrock_client: Optional BedrockClient instance for LLM-based recommendations
        """
        self.bedrock_client = bedrock_client
        logger.info("Dataset Recommender initialized")
    
    def recommend_datasets(self, paper: Dict[str, Any], 
                          paper_summary: str,
                          use_llm: bool = True) -> Dict[str, Any]:
        """
        Recommend datasets for a research paper.
        
        Args:
            paper: Paper document from OpenSearch
            paper_summary: Full paper summary from page classification (required)
            use_llm: Whether to use LLM for intelligent recommendation
            
        Returns:
            Dictionary with recommended datasets and reasoning
        """
        recommendations = {
            "paper_id": paper.get("_id"),
            "paper_title": paper.get("title", "Unknown"),
            "recommended_datasets": [],
            "available_datasets": list(AVAILABLE_DATASETS),
            "explicitly_mentioned": [],
            "domain_inferred": [],
            "reasoning": "",
            "confidence": "medium"
        }
        
        # Extract explicitly mentioned datasets
        explicitly_mentioned = self._extract_dataset_mentions(paper, paper_summary)
        recommendations["explicitly_mentioned"] = explicitly_mentioned
        
        # Use LLM for intelligent recommendation if available
        if use_llm and self.bedrock_client:
            llm_recommendations = self._llm_recommend_datasets(paper, paper_summary)
            if llm_recommendations:
                recommendations.update(llm_recommendations)
        
        # Combine and prioritize recommendations
        final_datasets = self._prioritize_datasets(
            explicitly_mentioned,
            recommendations.get("llm_recommended", [])
        )
        
        recommendations["recommended_datasets"] = final_datasets
        recommendations["primary_dataset"] = final_datasets[0] if final_datasets else "synthetic"
        
        return recommendations
    
    def _extract_dataset_mentions(self, paper: Dict[str, Any], 
                                  paper_summary: str) -> List[str]:
        """
        Extract explicitly mentioned datasets from paper text.
        
        Args:
            paper: Paper document
            paper_summary: Full paper summary from page classification (required)
            
        Returns:
            List of normalized dataset names
        """
        mentioned_datasets = set()
        
        # Use the full paper summary (limit to first 50k chars for performance)
        text_to_search = paper_summary[:50000]
        
        # Search for dataset patterns
        for pattern in DATASET_PATTERNS:
            matches = re.findall(pattern, text_to_search, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Handle capture groups
                normalized = self._normalize_dataset_name(match)
                if normalized:
                    mentioned_datasets.add(normalized)
        
        # Also check for lowercase mentions
        text_lower = text_to_search.lower()
        for dataset_name, normalized in DATASET_NAME_MAP.items():
            if dataset_name in text_lower:
                mentioned_datasets.add(normalized)
        
        return list(mentioned_datasets)
    
    def _normalize_dataset_name(self, dataset_name: str) -> Optional[str]:
        """
        Normalize dataset name to our standard format.
        
        Args:
            dataset_name: Raw dataset name from text
            
        Returns:
            Normalized dataset name or None
        """
        name_lower = dataset_name.lower().strip()
        return DATASET_NAME_MAP.get(name_lower, None)

    
    def _prioritize_datasets(self, explicitly_mentioned: List[str],
                            llm_recommended: List[str]) -> List[str]:
        """
        Prioritize and combine dataset recommendations.
        
        Args:
            explicitly_mentioned: Datasets explicitly mentioned in paper
            llm_recommended: Datasets recommended by LLM
            
        Returns:
            Prioritized list of dataset names
        """
        prioritized = []
        seen = set()
        
        # Priority 1: Explicitly mentioned datasets that we have
        for dataset in explicitly_mentioned:
            if dataset in AVAILABLE_DATASETS and dataset not in seen:
                prioritized.append(dataset)
                seen.add(dataset)
        
        # Priority 2: LLM recommendations
        for dataset in llm_recommended:
            if dataset in AVAILABLE_DATASETS and dataset not in seen:
                prioritized.append(dataset)
                seen.add(dataset)
    
        
        # Fallback to mnist if nothing found (simple, universal dataset)
        if not prioritized:
            prioritized = ["mnist"]
        
        return prioritized
    
    def _llm_recommend_datasets(self, paper: Dict[str, Any],
                                paper_summary: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to intelligently recommend datasets.
        
        Args:
            paper: Paper document
            paper_summary: Full paper summary from page classification (required)
            
        Returns:
            Dictionary with LLM recommendations
        """
        if not self.bedrock_client:
            return None
        
        try:
            # Use the full paper summary from page classification
            paper_info = f"""
Title: {paper.get('title', 'Unknown')}

Full Paper Summary (from page classification):
{paper_summary}
"""
            
            # Prepare prompt for dataset recommendation
            prompt = f"""
You are an expert in machine learning datasets. Analyze this research paper and recommend the most appropriate datasets for training and evaluating models described in the paper.

Paper Information:
{paper_info}

Available Datasets (ONLY THESE ARE AVAILABLE ON TRAINIUM):
- cifar10: 60K 32x32 color images, 10 classes (computer vision, image classification) - ✅ AVAILABLE
- cifar100: 60K 32x32 color images, 100 classes (computer vision, fine-grained classification) - ✅ AVAILABLE
- mnist: 70K 28x28 grayscale digits (simple vision tasks, baselines) - ✅ AVAILABLE
- fashion_mnist: 70K 28x28 grayscale fashion items (computer vision) - ✅ AVAILABLE
- imdb: 50K movie reviews for sentiment classification (NLP, text classification) - ✅ AVAILABLE
- wikitext2: 36K Wikipedia articles for language modeling (NLP, language modeling) - ✅ AVAILABLE
- synthetic: 16K synthetic samples for quick testing (various types: vision, tabular) - ✅ AVAILABLE

IMPORTANT: For NLP tasks, use proper NLP datasets (imdb, wikitext2) instead of vision datasets.

Based on the paper's domain, task type, and requirements, recommend 1-3 datasets from the available list above.

Consider:
1. What type of data does this paper work with? (images, text, etc.)
2. What is the primary task? (classification, language modeling, etc.)
3. What datasets would best demonstrate the paper's contributions?

Respond in JSON format:
{{
    "recommended_datasets": ["dataset1", "dataset2"],
    "primary_dataset": "dataset1",
    "reasoning": "Brief explanation of why these datasets are appropriate",
    "confidence": "high|medium|low"
}}

Only use datasets from the available list above. If none are perfect matches, choose the closest alternatives.
"""
            
            # Make LLM request
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_client.client.invoke_model(
                modelId=self.bedrock_client.model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            
            response_body = json.loads(response['body'].read())
            generated_text = response_body['content'][0]['text']
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', generated_text, re.DOTALL)
            if json_match:
                llm_result = json.loads(json_match.group())
                return {
                    "llm_recommended": llm_result.get("recommended_datasets", []),
                    "llm_reasoning": llm_result.get("reasoning", ""),
                    "llm_confidence": llm_result.get("confidence", "medium")
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in LLM dataset recommendation: {e}")
            return None
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        dataset_info = {
            "cifar10": {
                "name": "CIFAR-10",
                "type": "image classification",
                "samples": "60K",
                "classes": 10,
                "size": "32x32 RGB images",
                "use_case": "Computer vision, CNNs, image classification"
            },
            "cifar100": {
                "name": "CIFAR-100",
                "type": "image classification",
                "samples": "60K",
                "classes": 100,
                "size": "32x32 RGB images",
                "use_case": "Fine-grained classification, transfer learning"
            },
            "mnist": {
                "name": "MNIST",
                "type": "image classification",
                "samples": "70K",
                "classes": 10,
                "size": "28x28 grayscale images",
                "use_case": "Simple baselines, digit recognition"
            },
            "fashion_mnist": {
                "name": "Fashion-MNIST",
                "type": "image classification",
                "samples": "70K",
                "classes": 10,
                "size": "28x28 grayscale images",
                "use_case": "Computer vision, fashion classification"
            },
            "imdb": {
                "name": "IMDB",
                "type": "text classification",
                "samples": "50K",
                "classes": 2,
                "size": "Movie reviews (HuggingFace Arrow format)",
                "use_case": "NLP, sentiment analysis, text classification"
            },
            "wikitext2": {
                "name": "WikiText-2",
                "type": "language modeling",
                "samples": "10K+",
                "classes": "N/A",
                "size": "Wikipedia articles (HuggingFace Arrow format)",
                "use_case": "Language modeling, transformers, LLMs"
            },
            "synthetic": {
                "name": "Synthetic",
                "type": "various",
                "samples": "16K",
                "classes": "configurable",
                "size": "Generated data",
                "use_case": "Quick testing, debugging, prototyping"
            }
        }
        
        return dataset_info.get(dataset_name.lower(), {
            "name": dataset_name,
            "type": "unknown",
            "use_case": "Unknown dataset"
        })

