#!/bin/bash
set -e

FUNCTION_NAME="PapersScraper"
ZIP_FILE="scraper.zip"

echo "📦 Packaging $FUNCTION_NAME..."

# Move into scraper directory
cd ../scraper_lambda

# Install dependencies into a local folder
pip install -r ../deploy/requirements_scraper.txt -t .

# Zip everything
zip -r9 ../deploy/$ZIP_FILE .

# Clean up installed deps
find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf boto3* requests* selenium* bs4* dotenv*

cd ../deploy

echo "🚀 Updating $FUNCTION_NAME in AWS..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."
