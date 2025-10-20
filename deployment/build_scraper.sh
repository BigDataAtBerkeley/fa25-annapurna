#!/bin/bash

# Build and deploy scraper Lambda functions
FUNCTION_NAME=${1:-PaperScraper_ICLR}

echo "Building scraper Lambda for: $FUNCTION_NAME"

# Clean up old zip
rm -f scraper_lambda.zip

# Create zip from scraper_lambda directory
cd scraper_lambda
zip -r ../scraper_lambda.zip . -x "*.log" "logs/*" "__pycache__/*" "*.pyc" "*.zip"
cd ..

# Update Lambda function
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://scraper_lambda.zip

echo "Scraper Lambda $FUNCTION_NAME updated successfully"