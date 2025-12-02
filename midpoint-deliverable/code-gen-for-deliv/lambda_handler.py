"""
Lambda handler for Pipeline Code Generation.

This Lambda function:
1. Generates code using ChunkedPyTorchGenerator (PDF chunking with smart relevance filtering)
2. Sends initial generated code to the async Flask app's /execute endpoint for Trainium execution
3. Shuts down – all execution, code review, and retries happen in the async Flask app

This is the entry point for the pipeline Lambda. The Flask app handles the rest of the pipeline.
"""

import os
import json
import logging
import time
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import sys

sys.path.append('/opt/python')
sys.path.append(os.path.dirname(__file__))

from chunked_generator import ChunkedPyTorchGenerator  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Flask app endpoint for code execution
FLASK_EXECUTE_ENDPOINT = os.getenv('FLASK_EXECUTE_ENDPOINT')  # e.g., "http://1.2.3.4:8000/execute"
ENABLE_EXECUTION_TESTING = os.getenv('ENABLE_EXECUTION_TESTING', 'false').lower() == 'true'

def send_to_flask_app(paper_id: str, code: str, paper_title: Optional[str] = None, 
                      slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
    """
    Send reviewed code to Flask app's /execute endpoint for Trainium execution.
    
    Args:
        paper_id: Paper ID
        code: Reviewed code to execute
        paper_title: Paper title (optional)
        slack_thread_ts: Slack thread timestamp (optional)
        
    Returns:
        Response from Flask app
    """
    if not FLASK_EXECUTE_ENDPOINT:
        logger.warning("FLASK_EXECUTE_ENDPOINT not set - cannot send code to Flask app")
        return {
            "success": False,
            "error": "FLASK_EXECUTE_ENDPOINT not configured"
        }
    
    endpoint = FLASK_EXECUTE_ENDPOINT
    if not endpoint.endswith('/execute'):
        endpoint = f"{endpoint.rstrip('/')}/execute"
    
    payload = {
        "paper_id": paper_id,
        "code": code,
        "timeout": int(os.getenv('TRAINIUM_EXECUTION_TIMEOUT', '3600'))  # 60 minutes default
    }
    
    if paper_title:
        payload["paper_title"] = paper_title
    
    if slack_thread_ts:
        payload["slack_thread_ts"] = slack_thread_ts
    
    logger.info(f"Sending reviewed code to Flask app: {endpoint}")
    
    try:
        # Flask app returns 202 Accepted for async execution
        response = requests.post(
            endpoint,
            json=payload,
            timeout=30  # Short timeout - Flask app should accept quickly
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"✅ Code sent to Flask app successfully: {result.get('status', 'accepted')}")
        return {
            "success": True,
            "flask_response": result,
            "status": result.get("status", "accepted"),
            "job_id": result.get("job_id"),
            "status_url": result.get("status_url")
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send code to Flask app: {e}")
        return {
            "success": False,
            "error": f"Failed to send code to Flask app: {str(e)}"
        }


class PipelineHandler:
    """Main handler for the pipeline Lambda (code generation only)."""
    
    def __init__(self):
        """Initialize the pipeline handler."""
        # Initialize code generator (PDF chunking with smart relevance filtering)
        self.generator = ChunkedPyTorchGenerator(
            batch_size=8,  # Group 8 chunk summaries into each batch for hierarchical summarization
            use_smart_pdf_chunking=True,  # Enable smart chunking to prioritize relevant sections (filters appendix)
            max_pdf_chunks=15,  # Maximum number of PDF chunks to process (prioritizes abstract, formulas, diagrams)
            pages_per_pdf_chunk=2  # 2 pages per PDF chunk
        )

        logger.info("Pipeline Handler initialized (code generation only)")
    
    def process_paper(self, paper_id: str, slack_thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single paper: generate code and send it to the async Flask app.
        
        Args:
            paper_id: Paper ID to process
            slack_thread_ts: Optional Slack thread timestamp for notifications
            
        Returns:
            Result dictionary with generation, review, and Flask app response
        """
        start_time = time.time()
        result = {
            "paper_id": paper_id,
            "success": False,
            "pipeline_start": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Generate code
            logger.info(f"Step 1: Generating code for paper {paper_id}...")
            code_gen_result = self.generator.generate_code_for_paper(paper_id)
            
            if not code_gen_result.get("success") or not code_gen_result.get("code"):
                result["error"] = code_gen_result.get("error", "Code generation failed")
                result["code_generation"] = {
                    "success": False,
                    "error": code_gen_result.get("error")
                }
                return result
            
            logger.info(f"✅ Code generated ({len(code_gen_result['code']):,} chars)")
            result["code_generation"] = {
                "success": True,
                "code_length": len(code_gen_result["code"]),
                "model_used": code_gen_result.get("model_used"),
                "recommended_dataset": code_gen_result.get("recommended_dataset")
            }

            # Step 2: Send generated code to Flask app for execution
            logger.info("Step 2: Sending generated code to Flask app for execution...")
            flask_result = send_to_flask_app(
                paper_id=paper_id,
                code=code_gen_result["code"],
                paper_title=code_gen_result.get("paper_title"),
                slack_thread_ts=slack_thread_ts
            )
            
            result["flask_app"] = flask_result
            result["success"] = flask_result.get("success", False)
            result["pipeline_end"] = datetime.now().isoformat()
            result["total_pipeline_time"] = time.time() - start_time
            
            if result["success"]:
                logger.info(
                    f"✅ Pipeline complete for paper {paper_id} "
                    f"(total time: {result['total_pipeline_time']:.1f}s)"
                )
            else:
                logger.error(f"❌ Pipeline failed for paper {paper_id}: {flask_result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            result["error"] = str(e)
            result["pipeline_end"] = datetime.now().isoformat()
            result["total_pipeline_time"] = time.time() - start_time
            return result


def lambda_handler(event, context):
    """
    AWS Lambda handler function for pipeline (code generation only).
    
    This Lambda:
    1. Generates code using ChunkedPyTorchGenerator
    2. Sends initial code to Flask app's /execute endpoint
    3. Shuts down (execution, review, retries handled by async Flask app)
    
    Args:
        event: Lambda event with 'paper_id' (and optionally 'slack_thread_ts')
        context: Lambda context
        
    Returns:
        Response dictionary with pipeline results
    """
    try:
        logger.info("Pipeline Lambda invoked")
        
        handler = PipelineHandler()
        
        # Extract paper_id from event
        paper_id = event.get('paper_id')
        if not paper_id:
            return {
                "success": False,
                "error": "paper_id is required in event"
            }
        
        slack_thread_ts = event.get('slack_thread_ts')  # Optional
        
        # Process paper: generate code, review it, send to Flask app
        result = handler.process_paper(paper_id, slack_thread_ts=slack_thread_ts)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Pipeline Lambda: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e)
        }
