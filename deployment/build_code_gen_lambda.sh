#!/bin/bash

# Lambda deployment script for code generation system
# This script packages the code_gen module for AWS Lambda deployment

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

echo "Packaging Code Generation Lambda Function..."

# Create deployment directory
DEPLOY_DIR="lambda_deploy"
PACKAGE_NAME="code_gen_lambda.zip"

# Clean up previous builds
rm -rf $DEPLOY_DIR
rm -f $PACKAGE_NAME

# Create deployment directory
mkdir -p $DEPLOY_DIR

# Copy all Python files from code_gen to root (so imports work)
echo "Copying code_gen files..."
cp code_gen/*.py $DEPLOY_DIR/

# Install dependencies
echo "Installing dependencies..."
pip install -r code_gen/requirements.txt -t $DEPLOY_DIR/

# Don't need to create it, comment out the cat command
: << 'SKIP'
"""
AWS Lambda handler for the code generation system.
"""

import json
import logging
from code_gen.lambda_handler import lambda_handler as code_gen_handler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def lambda_handler(event, context):
    """
    AWS Lambda entry point for code generation.
    
    Args:
        event: Lambda event (dict)
        context: Lambda context
        
    Returns:
        Response dictionary
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Call the main code generation handler
        result = code_gen_handler(event, context)
        
        logger.info(f"Code generation result: {result.get('success', False)}")
        return result
        
    except Exception as e:
        logger.error(f"Lambda error: {str(e)}")
        return {
            "success": False,
            "error": f"Lambda error: {str(e)}",
            "event": event
        }
SKIP

# Create deployment package
echo "Creating deployment package..."
cd $DEPLOY_DIR
zip -r ../$PACKAGE_NAME . -x "*.pyc" "*/__pycache__/*" "*/tests/*" "*/test_*"
cd ..

echo "Package created: $PACKAGE_NAME"
echo "Package size: $(du -h $PACKAGE_NAME | cut -f1)"

# Clean up
rm -rf $DEPLOY_DIR

echo "Lambda package ready for deployment!"
echo ""
echo "Next steps:"
echo "1. Upload $PACKAGE_NAME to AWS Lambda"
echo "2. Set handler to: lambda_handler.lambda_handler"
echo "3. Set timeout to at least 5 minutes"
echo "4. Set memory to at least 1024 MB"
echo "5. Configure environment variables"
