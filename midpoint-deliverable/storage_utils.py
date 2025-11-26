#!/usr/bin/env python3
"""
Storage Utilities

Provides functions for saving JSON data and code files to the results directory.
Uses per-paper folder structure: results/{paper_id}/{step_name}/
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Results directory structure - per-paper folders
RESULTS_DIR = Path('results')


def save_json(paper_id: str, step_name: str, data: Dict[str, Any]) -> str:
    """
    Save JSON data to storage.
    Uses per-paper folder structure: results/{paper_id}/{step_name}/
    
    Args:
        paper_id: Paper ID
        step_name: Name of the step (e.g., 'code-generation', 'code-review', 'trn-execution')
        data: Data dictionary to save as JSON
        
    Returns:
        Path to saved file
    """
    # Create directory structure: results/{paper_id}/{step_name}/
    step_dir = RESULTS_DIR / paper_id / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp: {paper_id}_{timestamp}.json
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{paper_id}_{timestamp}.json"
    filepath = step_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise


def save_code(paper_id: str, step_name: str, code: str) -> str:
    """
    Save code to storage.
    Uses per-paper folder structure: results/{paper_id}/{step_name}/
    
    Args:
        paper_id: Paper ID
        step_name: Name of the step (e.g., 'code-generation', 'code-review')
        code: Code content to save
        
    Returns:
        Path to saved file
    """
    # Create directory structure: results/{paper_id}/{step_name}/
    step_dir = RESULTS_DIR / paper_id / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp: {paper_id}_{timestamp}.py
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{paper_id}_{timestamp}.py"
    filepath = step_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        logger.info(f"Saved code to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving code to {filepath}: {e}")
        raise

