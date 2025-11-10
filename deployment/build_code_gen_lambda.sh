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

echo "✅ Code deployed successfully."

# Update BEDROCK_MODEL_ID environment variable if it exists
echo ""
echo "🔄 Checking environment variables..."
CURRENT_ENV=$(aws lambda get-function-configuration \
  --function-name $FUNCTION_NAME \
  --query 'Environment.Variables' \
  --output json 2>/dev/null || echo "{}")

if [ "$CURRENT_ENV" != "{}" ] && [ "$CURRENT_ENV" != "null" ]; then
  # Check if BEDROCK_MODEL_ID exists and update it to Claude 3.5 Sonnet
  NEW_MODEL_ID="arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
  
  # Use Python to update the JSON (more reliable than requiring jq)
  ENV_STRING=$(python3 << EOF
import json
import sys

try:
    env_json = json.loads('''$CURRENT_ENV''')
    env_json['BEDROCK_MODEL_ID'] = '$NEW_MODEL_ID'
    env_string = ','.join([f"{k}={v}" for k, v in env_json.items()])
    print(env_string)
except Exception as e:
    print("", file=sys.stderr)
    sys.exit(1)
EOF
)
  
  if [ -n "$ENV_STRING" ]; then
    echo "🔄 Updating BEDROCK_MODEL_ID to Claude 3.5 Sonnet..."
    aws lambda update-function-configuration \
      --function-name $FUNCTION_NAME \
      --environment "Variables={$ENV_STRING}" > /dev/null
    
    echo "✅ Environment variable updated."
  else
    echo "⚠️  Failed to update environment variables. Update BEDROCK_MODEL_ID manually in Lambda console."
  fi
else
  echo "⚠️  No environment variables found. Set BEDROCK_MODEL_ID manually in Lambda console."
fi

echo ""
echo "✅ $FUNCTION_NAME deployed successfully."
echo ""
echo "📋 Configuration:"
echo "  Handler: lambda_handler.lambda_handler"
echo "  Timeout: 300 seconds (5 minutes)"
echo "  Memory: 1024 MB"
