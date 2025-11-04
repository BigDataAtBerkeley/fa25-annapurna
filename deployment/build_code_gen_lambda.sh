#!/bin/bash
set -e

# Build and deploy Code Generation Lambda function
FUNCTION_NAME="PapersCodeGenerator"
ZIP_FILE="code_gen_lambda.zip"

echo "📦 Packaging $FUNCTION_NAME..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Create temporary deployment directory
DEPLOY_DIR="lambda_deploy"

# Clean up previous builds
rm -rf $DEPLOY_DIR
rm -f deployment/$ZIP_FILE
rm -f $ZIP_FILE

# Create deployment directory
mkdir -p $DEPLOY_DIR

# Copy all Python files from code_gen
echo "📥 Copying code_gen files..."
cp code_gen/*.py $DEPLOY_DIR/

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r code_gen/requirements.txt -t $DEPLOY_DIR/

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
echo "  Handler: lambda_handler.lambda_handler"
echo "  Timeout: 300 seconds (5 minutes)"
echo "  Memory: 1024 MB"
