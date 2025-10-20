"""
Configuration for RAG-based paper deduplication system.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RAGConfig:
    """Configuration class for RAG system parameters."""
    
    # Embeddings Configuration
    EMBEDDINGS_MODEL_ID = os.getenv("EMBEDDINGS_MODEL_ID", "amazon.titan-embed-text-v1")
    EMBEDDING_DIMENSION = 1536  # Titan Embeddings v1 dimension
    
    # Similarity Thresholds - Adjusted for academic paper diversity
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))  # 95% similarity for redundancy (near-duplicates only)
    HIGH_SIMILARITY_THRESHOLD = float(os.getenv("HIGH_SIMILARITY_THRESHOLD", "0.98"))  # 98% for exact duplicates
    MEDIUM_SIMILARITY_THRESHOLD = float(os.getenv("MEDIUM_SIMILARITY_THRESHOLD", "0.90"))  # 90% for potential duplicates
    
    # Search Configuration
    DEFAULT_SEARCH_SIZE = int(os.getenv("DEFAULT_SEARCH_SIZE", "10"))
    MAX_SEARCH_SIZE = int(os.getenv("MAX_SEARCH_SIZE", "50"))
    NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", "100"))
    
    # OpenSearch k-NN Configuration
    KNN_EF_SEARCH = int(os.getenv("KNN_EF_SEARCH", "100"))
    KNN_EF_CONSTRUCTION = int(os.getenv("KNN_EF_CONSTRUCTION", "128"))
    KNN_M = int(os.getenv("KNN_M", "24"))
    
    # Processing Configuration
    BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "10"))
    TIMEOUT_SECONDS = int(os.getenv("RAG_TIMEOUT_SECONDS", "60"))
    
    @classmethod
    def get_similarity_thresholds(cls) -> Dict[str, float]:
        """Get all similarity thresholds."""
        return {
            "redundancy": cls.SIMILARITY_THRESHOLD,
            "high_similarity": cls.HIGH_SIMILARITY_THRESHOLD,
            "medium_similarity": cls.MEDIUM_SIMILARITY_THRESHOLD
        }
    
    @classmethod
    def get_opensearch_knn_settings(cls) -> Dict[str, Any]:
        """Get OpenSearch k-NN settings."""
        return {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": cls.KNN_EF_SEARCH
            }
        }
    
    @classmethod
    def get_knn_vector_mapping(cls) -> Dict[str, Any]:
        """Get k-NN vector field mapping."""
        return {
            "type": "knn_vector",
            "dimension": cls.EMBEDDING_DIMENSION,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "nmslib",
                "parameters": {
                    "ef_construction": cls.KNN_EF_CONSTRUCTION,
                    "m": cls.KNN_M
                }
            }
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate RAG configuration."""
        try:
            # Check similarity thresholds are valid
            thresholds = [cls.SIMILARITY_THRESHOLD, cls.HIGH_SIMILARITY_THRESHOLD, cls.MEDIUM_SIMILARITY_THRESHOLD]
            for threshold in thresholds:
                if not 0.0 <= threshold <= 1.0:
                    return False
            
            # Check search sizes are positive
            if cls.DEFAULT_SEARCH_SIZE <= 0 or cls.MAX_SEARCH_SIZE <= 0:
                return False
            
            # Check k-NN parameters
            if cls.KNN_EF_SEARCH <= 0 or cls.KNN_EF_CONSTRUCTION <= 0 or cls.KNN_M <= 0:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            "embeddings_model_id": cls.EMBEDDINGS_MODEL_ID,
            "embedding_dimension": cls.EMBEDDING_DIMENSION,
            "similarity_thresholds": cls.get_similarity_thresholds(),
            "search_config": {
                "default_search_size": cls.DEFAULT_SEARCH_SIZE,
                "max_search_size": cls.MAX_SEARCH_SIZE,
                "num_candidates": cls.NUM_CANDIDATES
            },
            "opensearch_knn": cls.get_opensearch_knn_settings(),
            "processing": {
                "batch_size": cls.BATCH_SIZE,
                "timeout_seconds": cls.TIMEOUT_SECONDS
            }
        }
