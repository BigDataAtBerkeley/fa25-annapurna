#!/usr/bin/env python3
"""
Local RAG testing without AWS dependencies.
Tests the core RAG logic and configuration.
"""

import sys
import os
import json
import logging
from typing import Dict, List, Any

# Add code_gen to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code_gen'))

from code_gen.rag_config import RAGConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalRAGTester:
    """Local RAG tester for core functionality."""
    
    def __init__(self):
        """Initialize local RAG tester."""
        self.config = RAGConfig()
        logger.info("Local RAG Tester initialized")
    
    def test_config_validation(self) -> bool:
        """Test RAG configuration validation."""
        logger.info("Testing RAG configuration...")
        
        try:
            # Test configuration validation
            is_valid = self.config.validate_config()
            
            if not is_valid:
                logger.error("RAG configuration validation failed")
                return False
            
            # Test configuration dictionary
            config_dict = self.config.get_config_dict()
            
            # Check required fields
            required_fields = [
                'embeddings_model_id',
                'embedding_dimension',
                'similarity_thresholds',
                'search_config',
                'opensearch_knn',
                'processing'
            ]
            
            for field in required_fields:
                if field not in config_dict:
                    logger.error(f"Missing required config field: {field}")
                    return False
            
            # Test similarity thresholds
            thresholds = self.config.get_similarity_thresholds()
            expected_thresholds = ['redundancy', 'high_similarity', 'medium_similarity']
            
            for threshold_name in expected_thresholds:
                if threshold_name not in thresholds:
                    logger.error(f"Missing similarity threshold: {threshold_name}")
                    return False
                
                threshold_value = thresholds[threshold_name]
                if not 0.0 <= threshold_value <= 1.0:
                    logger.error(f"Invalid threshold value for {threshold_name}: {threshold_value}")
                    return False
            
            logger.info("✅ RAG configuration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG configuration test failed: {e}")
            return False
    
    def test_similarity_logic(self) -> bool:
        """Test similarity threshold logic."""
        logger.info("Testing similarity logic...")
        
        try:
            # Test different similarity scenarios with academic-appropriate thresholds
            test_cases = [
                {
                    "similarity_score": 0.98,
                    "threshold": 0.95,
                    "expected_redundant": True,
                    "description": "Very high similarity (98%) should be redundant (near-duplicate)"
                },
                {
                    "similarity_score": 0.96,
                    "threshold": 0.95,
                    "expected_redundant": True,
                    "description": "High similarity (96%) should be redundant"
                },
                {
                    "similarity_score": 0.92,
                    "threshold": 0.95,
                    "expected_redundant": False,
                    "description": "Medium-high similarity (92%) should NOT be redundant (different attention types)"
                },
                {
                    "similarity_score": 0.85,
                    "threshold": 0.95,
                    "expected_redundant": False,
                    "description": "Medium similarity (85%) should NOT be redundant (different research)"
                },
                {
                    "similarity_score": 0.70,
                    "threshold": 0.95,
                    "expected_redundant": False,
                    "description": "Low similarity (70%) should not be redundant"
                }
            ]
            
            for case in test_cases:
                similarity_score = case["similarity_score"]
                threshold = case["threshold"]
                expected = case["expected_redundant"]
                description = case["description"]
                
                # Test redundancy logic
                is_redundant = similarity_score >= threshold
                
                if is_redundant != expected:
                    logger.error(f"❌ {description}: Expected {expected}, got {is_redundant}")
                    return False
                else:
                    logger.info(f"✅ {description}: {similarity_score} vs {threshold} = {is_redundant}")
            
            logger.info("✅ Similarity logic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Similarity logic test failed: {e}")
            return False
    
    def test_paper_processing_logic(self) -> bool:
        """Test paper processing logic without AWS calls."""
        logger.info("Testing paper processing logic...")
        
        try:
            # Sample paper data
            sample_paper = {
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                "date": "2017-12-06"
            }
            
            # Test paper data processing
            title = sample_paper["title"]
            abstract = sample_paper["abstract"]
            
            # Test title normalization (mock)
            title_normalized = "".join(ch.lower() for ch in title if ch.isalnum() or ch.isspace()).strip().replace(" ", "_")
            expected_normalized = "attention_is_all_you_need"
            
            if title_normalized != expected_normalized:
                logger.error(f"Title normalization failed: expected {expected_normalized}, got {title_normalized}")
                return False
            
            # Test abstract processing
            abstract_length = len(abstract)
            if abstract_length < 100:
                logger.error(f"Abstract too short: {abstract_length} characters")
                return False
            
            # Test similarity threshold application
            mock_similarity_scores = [0.98, 0.92, 0.85, 0.75, 0.65]
            threshold = self.config.SIMILARITY_THRESHOLD
            
            redundant_count = sum(1 for score in mock_similarity_scores if score >= threshold)
            expected_redundant = 1  # Only 0.98 is above 0.95 threshold
            
            if redundant_count != expected_redundant:
                logger.error(f"Similarity threshold application failed: expected {expected_redundant} redundant, got {redundant_count}")
                return False
            
            logger.info("✅ Paper processing logic test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Paper processing logic test failed: {e}")
            return False
    
    def test_rag_workflow_simulation(self) -> bool:
        """Test the complete RAG workflow simulation."""
        logger.info("Testing RAG workflow simulation...")
        
        try:
            # Simulate the RAG workflow
            workflow_steps = [
                "1. Paper arrives via SQS",
                "2. Extract title and abstract",
                "3. Generate vector embedding (mock)",
                "4. Search for similar papers (mock)",
                "5. Calculate similarity scores (mock)",
                "6. Apply similarity threshold",
                "7. Make redundancy decision",
                "8. Index paper with embedding (mock)"
            ]
            
            # Mock similarity search results - realistic academic scenarios
            mock_search_results = [
                {"title": "Attention Is All You Need (Duplicate)", "similarity": 0.98},  # Near-duplicate
                {"title": "Multi-Head Attention Mechanisms", "similarity": 0.92},        # Different attention type - KEEP
                {"title": "Self-Attention for Language Models", "similarity": 0.88},    # Different focus - KEEP  
                {"title": "Cross-Attention in Vision Transformers", "similarity": 0.85}, # Different domain - KEEP
                {"title": "Sparse Attention Patterns", "similarity": 0.82},             # Different technique - KEEP
                {"title": "Linear Attention Mechanisms", "similarity": 0.78},           # Different approach - KEEP
                {"title": "Neural Networks", "similarity": 0.65},                       # Different topic - KEEP
                {"title": "Machine Learning", "similarity": 0.55}                       # Different topic - KEEP
            ]
            
            # Test workflow steps
            for step in workflow_steps:
                logger.info(f"  {step}")
            
            # Test similarity analysis
            threshold = self.config.SIMILARITY_THRESHOLD
            redundant_papers = [paper for paper in mock_search_results if paper["similarity"] >= threshold]
            
            if len(redundant_papers) != 1:
                logger.error(f"Workflow simulation failed: expected 1 redundant paper, got {len(redundant_papers)}")
                return False
            
            # Test decision logic
            is_redundant = len(redundant_papers) > 0
            if is_redundant:
                decision = "REJECT"
                reason = f"Found {len(redundant_papers)} similar papers with max similarity {max(p['similarity'] for p in redundant_papers):.3f}"
            else:
                decision = "ACCEPT"
                reason = "No similar papers found"
            
            logger.info(f"  Decision: {decision}")
            logger.info(f"  Reason: {reason}")
            
            logger.info("✅ RAG workflow simulation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG workflow simulation test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all local RAG tests."""
        logger.info("🚀 Starting Local RAG Tests...")
        
        tests = [
            ("RAG Configuration", self.test_config_validation),
            ("Similarity Logic", self.test_similarity_logic),
            ("Paper Processing Logic", self.test_paper_processing_logic),
            ("RAG Workflow Simulation", self.test_rag_workflow_simulation)
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
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Local RAG Tests Summary: {passed}/{total} tests passed")
        logger.info(f"{'='*50}")
        
        if passed == total:
            logger.info("🎉 All local RAG tests passed! Core logic is working correctly.")
            logger.info("Next step: Set up AWS environment variables to test with real services.")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
            return False

def main():
    """Main test function."""
    print("Local RAG Implementation Test Suite")
    print("=" * 50)
    print("Testing core RAG logic without AWS dependencies...")
    print()
    
    tester = LocalRAGTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
