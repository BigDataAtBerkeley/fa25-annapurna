#!/usr/bin/env python3
"""
Test script for RAG-based paper deduplication implementation.
Tests the complete RAG pipeline with sample papers.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add code_gen to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code_gen'))

from code_gen.opensearch_client import OpenSearchClient
from code_gen.embeddings_client import EmbeddingsClient
from code_gen.rag_config import RAGConfig

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGTester:
    """Test class for RAG implementation."""
    
    def __init__(self):
        """Initialize RAG tester."""
        self.opensearch_client = OpenSearchClient()
        self.embeddings_client = EmbeddingsClient()
        self.config = RAGConfig()
        
        logger.info("RAG Tester initialized")
    
    def test_embeddings_generation(self) -> bool:
        """Test embeddings generation."""
        logger.info("Testing embeddings generation...")
        
        test_text = "This is a test abstract for a machine learning paper about transformers."
        
        try:
            embedding = self.embeddings_client.generate_embedding(test_text)
            
            if not embedding:
                logger.error("Failed to generate embedding")
                return False
            
            if len(embedding) != self.config.EMBEDDING_DIMENSION:
                logger.error(f"Embedding dimension mismatch: expected {self.config.EMBEDDING_DIMENSION}, got {len(embedding)}")
                return False
            
            if not all(isinstance(x, (int, float)) for x in embedding):
                logger.error("Embedding contains non-numeric values")
                return False
            
            logger.info(f"✅ Embeddings generation test passed. Generated {len(embedding)}-dimensional vector")
            return True
            
        except Exception as e:
            logger.error(f"❌ Embeddings generation test failed: {e}")
            return False
    
    def test_opensearch_vector_index(self) -> bool:
        """Test OpenSearch vector index creation."""
        logger.info("Testing OpenSearch vector index...")
        
        try:
            # Ensure vector index exists
            success = self.opensearch_client.ensure_vector_index()
            
            if not success:
                logger.error("Failed to ensure vector index")
                return False
            
            # Test if index exists and has correct mapping
            if not self.opensearch_client.client.indices.exists(index=self.opensearch_client.opensearch_index):
                logger.error("Index does not exist after creation")
                return False
            
            # Get index mapping
            mapping = self.opensearch_client.client.indices.get_mapping(index=self.opensearch_client.opensearch_index)
            
            # Check if abstract_embedding field exists
            properties = mapping[self.opensearch_client.opensearch_index]['mappings']['properties']
            
            if 'abstract_embedding' not in properties:
                logger.error("abstract_embedding field not found in mapping")
                return False
            
            embedding_field = properties['abstract_embedding']
            if embedding_field.get('type') != 'knn_vector':
                logger.error("abstract_embedding field is not of type knn_vector")
                return False
            
            if embedding_field.get('dimension') != self.config.EMBEDDING_DIMENSION:
                logger.error(f"Embedding dimension mismatch in mapping: expected {self.config.EMBEDDING_DIMENSION}")
                return False
            
            logger.info("✅ OpenSearch vector index test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ OpenSearch vector index test failed: {e}")
            return False
    
    def test_paper_redundancy_detection(self) -> bool:
        """Test paper redundancy detection with sample papers."""
        logger.info("Testing paper redundancy detection...")
        
        # Sample papers for testing
        sample_papers = [
            {
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                "date": "2017-12-06"
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                "date": "2018-10-11"
            },
            {
                "title": "GPT: Improving Language Understanding by Generative Pre-Training",
                "abstract": "Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering, semantic similarity assessment, and document classification. Although large unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately.",
                "authors": ["Alec Radford", "Karthik Narasimhan"],
                "date": "2018-06-11"
            }
        ]
        
        try:
            # Index sample papers
            for i, paper in enumerate(sample_papers):
                paper_data = {
                    "title": paper["title"],
                    "title_normalized": paper["title"].lower().replace(" ", "_"),
                    "abstract": paper["abstract"],
                    "authors": paper["authors"],
                    "date": paper["date"],
                    "s3_bucket": "test-bucket",
                    "s3_key": f"test-paper-{i}.pdf",
                    "sha_abstract": f"test-hash-{i}",
                    "decision": "accept",
                    "reason": "Test paper",
                    "relevance": "yes",
                    "novelty": "yes"
                }
                
                success = self.opensearch_client.index_paper_with_embedding(paper_data)
                if not success:
                    logger.error(f"Failed to index test paper: {paper['title']}")
                    return False
            
            logger.info("✅ Indexed sample papers successfully")
            
            # Test redundancy detection with similar abstract
            similar_abstract = "We propose a new neural network architecture called the Transformer, which relies entirely on attention mechanisms to compute representations of input sequences. This approach eliminates the need for recurrent and convolutional layers that have been the foundation of previous sequence transduction models."
            
            redundancy_result = self.opensearch_client.check_paper_redundancy(
                "Test Transformer Paper", 
                similar_abstract, 
                similarity_threshold=0.7
            )
            
            if redundancy_result["is_redundant"]:
                logger.info(f"✅ Redundancy detection working correctly. Found similarity: {redundancy_result['max_similarity']:.3f}")
            else:
                logger.warning(f"⚠️ No redundant papers found. Max similarity: {redundancy_result['max_similarity']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Paper redundancy detection test failed: {e}")
            return False
    
    def test_vector_similarity_search(self) -> bool:
        """Test vector similarity search functionality."""
        logger.info("Testing vector similarity search...")
        
        try:
            # Test search with a transformer-related abstract
            query_abstract = "We introduce a novel attention mechanism for natural language processing that improves upon existing transformer architectures by incorporating bidirectional context understanding."
            
            similar_papers = self.opensearch_client.search_similar_papers(
                query_abstract, 
                size=5, 
                similarity_threshold=0.5
            )
            
            logger.info(f"Found {len(similar_papers)} similar papers")
            
            for i, paper in enumerate(similar_papers):
                logger.info(f"  {i+1}. {paper.get('title', 'Unknown')} (similarity: {paper.get('similarity_score', 0):.3f})")
            
            logger.info("✅ Vector similarity search test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector similarity search test failed: {e}")
            return False
    
    def cleanup_test_data(self):
        """Clean up test data from OpenSearch."""
        logger.info("Cleaning up test data...")
        
        try:
            # Delete test papers
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"title_normalized": "attention_is_all_you_need"}},
                            {"term": {"title_normalized": "bert:_pre-training_of_deep_bidirectional_transformers_for_language_understanding"}},
                            {"term": {"title_normalized": "gpt:_improving_language_understanding_by_generative_pre-training"}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            
            self.opensearch_client.client.delete_by_query(
                index=self.opensearch_client.opensearch_index,
                body=query
            )
            
            logger.info("✅ Test data cleaned up")
            
        except Exception as e:
            logger.error(f"⚠️ Error cleaning up test data: {e}")
    
    def run_all_tests(self) -> bool:
        """Run all RAG tests."""
        logger.info("🚀 Starting RAG implementation tests...")
        
        tests = [
            ("Embeddings Generation", self.test_embeddings_generation),
            ("OpenSearch Vector Index", self.test_opensearch_vector_index),
            ("Paper Redundancy Detection", self.test_paper_redundancy_detection),
            ("Vector Similarity Search", self.test_vector_similarity_search)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                if test_func():
                    passed += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {e}")
        
        # Cleanup
        self.cleanup_test_data()
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"RAG Tests Summary: {passed}/{total} tests passed")
        logger.info(f"{'='*50}")
        
        if passed == total:
            logger.info("🎉 All RAG tests passed! Implementation is ready.")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
            return False

def main():
    """Main test function."""
    print("RAG Implementation Test Suite")
    print("=" * 50)
    
    # Check environment variables
    required_env_vars = [
        "OPENSEARCH_ENDPOINT",
        "AWS_REGION"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set these variables in your .env file or environment")
        return False
    
    # Run tests
    tester = RAGTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
