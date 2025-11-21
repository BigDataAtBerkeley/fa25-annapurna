"""
Error Database Module

Stores execution errors for each paper_id in a simple file-based database.
For production, consider using DynamoDB or RDS.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Database directory
ERROR_DB_DIR = Path(os.getenv('ERROR_DB_DIR', '/tmp/trainium_errors'))
ERROR_DB_DIR.mkdir(parents=True, exist_ok=True)


def save_error(paper_id: str, error_data: Dict[str, Any]) -> str:
    """
    Save error to database.
    
    Args:
        paper_id: Paper/document ID
        error_data: Error information dictionary
        
    Returns:
        Path to saved error file
    """
    error_file = ERROR_DB_DIR / f"{paper_id}_errors.json"
    
    # Load existing errors if file exists
    errors = []
    if error_file.exists():
        try:
            with open(error_file, 'r') as f:
                data = json.load(f)
                errors = data.get('errors', [])
        except Exception as e:
            logger.warning(f"Failed to load existing errors: {e}")
            errors = []
    
    # Add new error with timestamp
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_data": error_data
    }
    errors.append(error_entry)
    
    # Save updated errors
    db_data = {
        "paper_id": paper_id,
        "last_updated": datetime.now().isoformat(),
        "error_count": len(errors),
        "errors": errors
    }
    
    with open(error_file, 'w') as f:
        json.dump(db_data, f, indent=2)
    
    logger.info(f"Saved error for {paper_id} (total errors: {len(errors)})")
    return str(error_file)


def get_errors(paper_id: str) -> List[Dict[str, Any]]:
    """
    Get all errors for a paper_id.
    
    Args:
        paper_id: Paper/document ID
        
    Returns:
        List of error dictionaries
    """
    error_file = ERROR_DB_DIR / f"{paper_id}_errors.json"
    
    if not error_file.exists():
        return []
    
    try:
        with open(error_file, 'r') as f:
            data = json.load(f)
            return data.get('errors', [])
    except Exception as e:
        logger.error(f"Failed to load errors for {paper_id}: {e}")
        return []


def get_latest_error(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent error for a paper_id.
    
    Args:
        paper_id: Paper/document ID
        
    Returns:
        Most recent error dictionary or None
    """
    errors = get_errors(paper_id)
    if errors:
        return errors[-1].get('error_data')
    return None


def clear_errors(paper_id: str) -> bool:
    """
    Clear all errors for a paper_id.
    
    Args:
        paper_id: Paper/document ID
        
    Returns:
        True if cleared, False otherwise
    """
    error_file = ERROR_DB_DIR / f"{paper_id}_errors.json"
    
    if error_file.exists():
        try:
            error_file.unlink()
            logger.info(f"Cleared errors for {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear errors for {paper_id}: {e}")
            return False
    return True


def get_error_count(paper_id: str) -> int:
    """Get number of errors for a paper_id."""
    return len(get_errors(paper_id))

