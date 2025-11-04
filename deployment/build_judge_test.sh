#!/bin/bash
set -e

# Build and deploy Judge Test Lambda function
FUNCTION_NAME="PaperJudgeTest"
ZIP_FILE="judge_test.zip"

echo "📦 Packaging $FUNCTION_NAME..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Move into judge_test directory
cd judge_lambda_test

# Clean old zip files
rm -f ../deployment/$ZIP_FILE
rm -f $ZIP_FILE

# Install dependencies into a local folder
echo "📥 Installing dependencies..."
pip install -r requirements.txt -t .

# Zip everything (excluding cache & logs)
echo "🗜️ Creating deployment package..."
zip -r9 ../deployment/$ZIP_FILE . -x "*.log" "logs/*" "__pycache__/*" "*.pyc" "*.zip"

# Clean up installed deps
echo "🧹 Cleaning up temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + >/dev/null 2>&1 || true
rm -rf boto3* opensearch_py* requests* certifi* botocore* s3transfer* urllib3* charset_normalizer* idna* six* python_dateutil* Events* jmespath* requests_aws4auth* || true

cd ../deployment

echo "🚀 Updating $FUNCTION_NAME in AWS..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."
