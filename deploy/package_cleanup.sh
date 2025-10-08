#!/bin/bash
set -e

FUNCTION_NAME="LogCleanupLambda"
ZIP_FILE="cleanup.zip"

echo "📦 Packaging $FUNCTION_NAME..."

cd ../cleanup_lambda
pip install -r ../deploy/requirements_cleanup.txt -t .

zip -r9 ../deploy/$ZIP_FILE .

find . -type d -name "__pycache__" -exec rm -rf {} +
rm -rf boto3* botocore* s3transfer*

cd ../deploy

echo "🚀 Updating $FUNCTION_NAME in AWS..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."
