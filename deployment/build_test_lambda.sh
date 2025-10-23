#!/bin/bash

# Lambda deployment script for code testing system
# This script packages the test_lambda for AWS Lambda deployment

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

echo "Packaging Code Testing Lambda Function..."

# Create deployment directory
DEPLOY_DIR="lambda_test_deploy"
PACKAGE_NAME="code_test_lambda.zip"

# Clean up previous builds
rm -rf $DEPLOY_DIR
rm -f $PACKAGE_NAME

# Create deployment directory
mkdir -p $DEPLOY_DIR

# Copy the lambda function
cp test_lambda/lambda_function.py $DEPLOY_DIR/
cp test_lambda/requirements.txt $DEPLOY_DIR/

# Install dependencies
echo "Installing dependencies..."
pip install -r test_lambda/requirements.txt -t $DEPLOY_DIR/

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
echo "2. Set handler to: lambda_function.lambda_handler"
echo "3. Set timeout to at least 15 minutes (900 seconds)"
echo "4. Set memory to at least 512 MB"
echo "5. Configure environment variables:"
echo "   - OPENSEARCH_ENDPOINT"
echo "   - OPENSEARCH_INDEX=research-papers-v2"
echo "   - OUTPUTS_BUCKET=papers-test-outputs"
echo "   - TRAINIUM_ENDPOINT (e.g., http://10.0.1.50:8000)"
echo "   - TRAINIUM_INSTANCE_ID (EC2 instance ID)"
echo "   - BATCH_SIZE=10"
echo "   - TRAINIUM_TIMEOUT=600"
