#!/bin/bash
set -e

# Build and deploy Code Testing Lambda function
FUNCTION_NAME="PapersCodeTester"
ZIP_FILE="code_test_lambda.zip"

echo "📦 Packaging $FUNCTION_NAME..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Create temporary deployment directory
DEPLOY_DIR="lambda_test_deploy"

# Clean up previous builds
rm -rf $DEPLOY_DIR
rm -f deployment/$ZIP_FILE
rm -f $ZIP_FILE

# Create deployment directory
mkdir -p $DEPLOY_DIR

# Copy the lambda function
echo "📥 Copying test_lambda files..."
cp test_lambda/lambda_function.py $DEPLOY_DIR/
cp test_lambda/requirements.txt $DEPLOY_DIR/

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r test_lambda/requirements.txt -t $DEPLOY_DIR/

# Create deployment package
echo "🗜️ Creating deployment package..."
cd $DEPLOY_DIR
zip -r9 ../deployment/$ZIP_FILE . -x "*.pyc" "*/__pycache__/*" "*/tests/*" "*/test_*"
cd ..

# Clean up
echo "🧹 Cleaning up temporary files..."
rm -rf $DEPLOY_DIR

echo "Package created: deployment/$ZIP_FILE"
echo "Package size: $(du -h deployment/$ZIP_FILE | cut -f1)"

cd deployment

echo "🚀 Updating $FUNCTION_NAME in AWS..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."
echo ""
echo "📋 Configuration:"
echo "  Handler: lambda_function.lambda_handler"
echo "  Timeout: 900 seconds (15 minutes)"
echo "  Memory: 512 MB"
echo "  Environment Variables:"
echo "    - OPENSEARCH_ENDPOINT"
echo "    - OPENSEARCH_INDEX=research-papers-v2"
echo "    - OUTPUTS_BUCKET=papers-test-outputs"
echo "    - TRAINIUM_ENDPOINT"
echo "    - TRAINIUM_INSTANCE_ID"
echo "    - BATCH_SIZE=10"
echo "    - TRAINIUM_TIMEOUT=600"
