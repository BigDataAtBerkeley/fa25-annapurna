#!/bin/bash
set -e

# Build and deploy Cleanup Lambda function
FUNCTION_NAME="LogCleanupLambda"
ZIP_FILE="cleanup.zip"

echo "📦 Packaging $FUNCTION_NAME..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Move into cleanup directory
cd cleanup_lambda

# Clean old zip files
rm -f ../deployment/$ZIP_FILE
rm -f $ZIP_FILE

# Install dependencies into a local folder
echo "📥 Installing dependencies..."
pip install -r requirements_cleanup.txt -t .

# Zip everything (excluding cache & logs)
echo "🗜️ Creating deployment package..."
zip -r9 ../deployment/$ZIP_FILE . -x "*.log" "logs/*" "__pycache__/*" "*.pyc" "*.zip"

# Clean up installed deps
echo "🧹 Cleaning up temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + >/dev/null 2>&1 || true
rm -rf boto3* botocore* s3transfer* urllib3* certifi* charset_normalizer* idna* six* python_dateutil* Events* jmespath* || true

cd ../deployment

echo "🚀 Updating $FUNCTION_NAME in AWS..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."