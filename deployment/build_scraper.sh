#!/bin/bash
set -e

# Build and deploy scraper Lambda functions
FUNCTION_NAME=${1:-PaperScraper_ICLR}
ZIP_FILE="scraper_lambda.zip"

echo "📦 Packaging $FUNCTION_NAME..."

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Move into scraper directory
cd scraper_lambda

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
rm -rf boto3* requests* selenium* bs4* dotenv* botocore* s3transfer* urllib3* charset_normalizer* idna* certifi* python_dateutil* || true

cd ../deployment

echo "🚀 Updating $FUNCTION_NAME in AWS..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."