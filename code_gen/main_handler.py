"""
Invokes Lambda to run pytorch_generator.py
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List
from .pytorch_generator import PyTorchCodeGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeGenHandler:
    """Main handler for the code generation system."""
    
    def __init__(self):
        """Initialize the code generation handler."""
        self.generator = PyTorchCodeGenerator()
        logger.info("Code Generation Handler initialized")
    
    def handle_lambda_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle AWS Lambda events for code generation.
        
        Args:
            event: Lambda event dictionary
            
        Returns:
            Response dictionary
        """
        try:
            # Extract parameters from event
            action = event.get('action', 'generate_by_id')
            paper_id = event.get('paper_id')
            paper_ids = event.get('paper_ids', [])
            title = event.get('title')
            author = event.get('author')
            keywords = event.get('keywords')
            max_papers = event.get('max_papers', 5)
            include_full_content = event.get('include_full_content', False)
            days = event.get('days', 30)
            
            logger.info(f"Processing action: {action}")
            
            # Route to appropriate handler
            if action == 'generate_by_id':
                if not paper_id:
                    return {"error": "paper_id is required for generate_by_id action"}
                result = self.generator.generate_code_for_paper(paper_id, include_full_content)
                
            elif action == 'generate_by_ids':
                if not paper_ids:
                    return {"error": "paper_ids is required for generate_by_ids action"}
                result = self.generator.generate_code_for_papers(paper_ids, include_full_content)
                
            elif action == 'generate_by_title':
                if not title:
                    return {"error": "title is required for generate_by_title action"}
                result = self.generator.generate_code_by_title(title, max_papers, include_full_content)
                
            elif action == 'generate_by_author':
                if not author:
                    return {"error": "author is required for generate_by_author action"}
                result = self.generator.generate_code_by_author(author, max_papers, include_full_content)
                
            elif action == 'generate_by_keywords':
                if not keywords:
                    return {"error": "keywords is required for generate_by_keywords action"}
                result = self.generator.generate_code_by_keywords(keywords, max_papers, include_full_content)
                
            elif action == 'generate_recent':
                result = self.generator.generate_code_for_recent_papers(days, max_papers, include_full_content)
                
            elif action == 'get_paper_info':
                if not paper_id:
                    return {"error": "paper_id is required for get_paper_info action"}
                result = self.generator.get_paper_info(paper_id)
                
            else:
                return {"error": f"Unknown action: {action}"}
            
            # Add metadata to response
            result.update({
                "action": action,
                "timestamp": result.get("generated_at", "unknown")
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling lambda event: {e}")
            return {
                "error": f"Error: {str(e)}",
                "action": event.get('action', 'unknown')
            }
    
    def run_cli(self, args: List[str] = None) -> None:
        """
        Run the code generator from command line.
        
        Args:
            args: Command line arguments
        """
        parser = argparse.ArgumentParser(description='Generate PyTorch code from research papers')
        
        # Action selection
        parser.add_argument('action', choices=[
            'generate_by_id', 'generate_by_ids', 'generate_by_title', 
            'generate_by_author', 'generate_by_keywords', 'generate_recent', 'get_paper_info'
        ], help='Action to perform')
        
        # Parameters
        parser.add_argument('--paper-id', help='Paper ID for single paper operations')
        parser.add_argument('--paper-ids', nargs='+', help='List of paper IDs')
        parser.add_argument('--title', help='Paper title to search for')
        parser.add_argument('--author', help='Author name to search for')
        parser.add_argument('--keywords', help='Keywords to search in abstract')
        parser.add_argument('--max-papers', type=int, default=5, help='Maximum number of papers to process')
        parser.add_argument('--include-full-content', action='store_true', help='Include full paper content')
        parser.add_argument('--days', type=int, default=30, help='Days to look back for recent papers')
        parser.add_argument('--output-dir', default='generated_code', help='Output directory for generated code')
        parser.add_argument('--save', action='store_true', help='Save generated code to files')
        
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        try:
            # Convert to event format
            event = {
                'action': parsed_args.action,
                'paper_id': parsed_args.paper_id,
                'paper_ids': parsed_args.paper_ids,
                'title': parsed_args.title,
                'author': parsed_args.author,
                'keywords': parsed_args.keywords,
                'max_papers': parsed_args.max_papers,
                'include_full_content': parsed_args.include_full_content,
                'days': parsed_args.days
            }
            
            # Process the request
            result = self.handle_lambda_event(event)
            
            # Print results
            print(json.dumps(result, indent=2))
            
            # Save code if requested
            if parsed_args.save and result.get('success'):
                if 'results' in result:
                    # Multiple results
                    for res in result['results']:
                        if res.get('success'):
                            self.generator.save_generated_code(res, parsed_args.output_dir)
                else:
                    # Single result
                    self.generator.save_generated_code(result, parsed_args.output_dir)
            
        except Exception as e:
            logger.error(f"Error in CLI: {e}")
            print(f"Error: {e}")

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    handler = CodeGenHandler()
    return handler.handle_lambda_event(event)

def main():
    """Main entry point for CLI usage."""
    handler = CodeGenHandler()
    handler.run_cli()

if __name__ == "__main__":
    main()
