#!/bin/bash
set -e

# Build and deploy Conference Wrapper Lambda function
FUNCTION_NAME="conferenceWrapper"
ZIP_FILE="conference_wrapper.zip"

echo "📦 Packaging $FUNCTION_NAME..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Move into conference_wrapper directory
cd conference_wrapper

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
# Note: boto3 is provided by AWS Lambda runtime
echo "🧹 Cleaning up temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + >/dev/null 2>&1 || true
# Remove boto3/botocore (provided by Lambda runtime) and OpenSearch (not needed)
# Keep: requests, beautifulsoup4, selenium and their dependencies (needed for web scraping)
rm -rf boto3* botocore* s3transfer* opensearch_py* Events* jmespath* requests_aws4auth* || true

cd ../deployment

echo "🚀 Updating $FUNCTION_NAME in AWS..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."

