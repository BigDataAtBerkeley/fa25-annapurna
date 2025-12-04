import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
    OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v3")
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
    S3_BUCKET = os.getenv("BUCKET_NAME")
    DEFAULT_MAX_PAPERS = 5
    DEFAULT_DAYS_BACK = 30
    DEFAULT_OUTPUT_DIR = "generated_code"
    
    @classmethod
    def validate(cls) -> bool:
        """
        just validates that all required env varis are present (returns true if all are present, false otherwise)
        """

        required_vars = [
            "OPENSEARCH_ENDPOINT",
            "BEDROCK_MODEL_ID"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            logging.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        return True
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        return {
            "aws_region": cls.AWS_REGION,
            "opensearch_endpoint": cls.OPENSEARCH_ENDPOINT,
            "opensearch_index": cls.OPENSEARCH_INDEX,
            "bedrock_model_id": cls.BEDROCK_MODEL_ID,
            "s3_bucket": cls.S3_BUCKET,
            "default_max_papers": cls.DEFAULT_MAX_PAPERS,
            "default_days_back": cls.DEFAULT_DAYS_BACK,
            "default_output_dir": cls.DEFAULT_OUTPUT_DIR
        }

class Logger:

    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

class FileUtils:
    @staticmethod
    def ensure_directory(path: str) -> None:
        """
        Ensures a directory exists, create if it doesn't.
        """
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> bool:
        """
        Save data as JSON file.
        
        Args:
            data: Data to save
            filepath: File path
            
        Returns True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Error saving JSON to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            filepath: File path
        Returns loaded data or None if error
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON from {filepath}: {e}")
            return None
    
    @staticmethod
    def save_code(code: str, filepath: str) -> bool:
        """
        Save code to file.
        
        Args:
            code: Code content
            filepath: File path
            
        Returns True if successful, False otherwise
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            return True
        except Exception as e:
            logging.error(f"Error saving code to {filepath}: {e}")
            return False

class ValidationUtils:
    """Validation utilities."""
    
    @staticmethod
    def validate_paper_id(paper_id: str) -> bool:
        """
        Validate paper ID format.
        
        Args:
            paper_id: Paper ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not paper_id or not isinstance(paper_id, str):
            return False
        
        # Basic validation - should be non-empty string
        return len(paper_id.strip()) > 0
    
    @staticmethod
    def validate_search_query(query: Dict[str, Any]) -> bool:
        """
        Validate OpenSearch query format.
        
        Args:
            query: Query to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(query, dict):
            return False
        
        # Check for common query structures
        valid_keys = ['match', 'match_all', 'range', 'bool', 'term', 'terms']
        return any(key in query for key in valid_keys)
    
    @staticmethod
    def validate_max_papers(max_papers: int) -> bool:
        """
        Validate max papers parameter.
        
        Args:
            max_papers: Number to validate
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(max_papers, int) and 1 <= max_papers <= 100

class ErrorHandler:
    """Error handling utilities."""
    
    @staticmethod
    def handle_opensearch_error(error: Exception) -> Dict[str, Any]:
        """
        Handle OpenSearch errors.
        
        Args:
            error: Exception to handle
            
        Returns:
            Error response dictionary
        """
        error_msg = str(error)
        
        if "ConnectionError" in error_msg:
            return {
                "error_type": "connection_error",
                "message": "Failed to connect to OpenSearch",
                "details": error_msg
            }
        elif "NotFoundError" in error_msg:
            return {
                "error_type": "not_found",
                "message": "Document not found",
                "details": error_msg
            }
        else:
            return {
                "error_type": "opensearch_error",
                "message": "OpenSearch error occurred",
                "details": error_msg
            }
    
    @staticmethod
    def handle_bedrock_error(error: Exception) -> Dict[str, Any]:
        """
        Handle Bedrock errors.
        
        Args:
            error: Exception to handle
            
        Returns:
            Error response dictionary
        """
        error_msg = str(error)
        
        if "ClientError" in error_msg:
            return {
                "error_type": "bedrock_client_error",
                "message": "Bedrock client error",
                "details": error_msg
            }
        elif "ModelError" in error_msg:
            return {
                "error_type": "model_error",
                "message": "Model error occurred",
                "details": error_msg
            }
        else:
            return {
                "error_type": "bedrock_error",
                "message": "Bedrock error occurred",
                "details": error_msg
            }
