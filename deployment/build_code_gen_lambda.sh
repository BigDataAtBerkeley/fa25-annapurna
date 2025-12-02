#!/bin/bash
set -e

# Build and deploy Pipeline Lambda function (code generation + review)
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

# Copy all Python files from code-gen-for-deliv
echo "📥 Copying code-gen-for-deliv files..."
cp midpoint-deliverable/code-gen-for-deliv/*.py $DEPLOY_DIR/

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r midpoint-deliverable/code-gen-for-deliv/requirements.txt -t $DEPLOY_DIR/

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
  # Use Claude 3 Haiku (as configured in the code)
  NEW_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"
  
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
    echo "🔄 Updating BEDROCK_MODEL_ID to Claude 3 Haiku..."
    aws lambda update-function-configuration \
      --function-name $FUNCTION_NAME \
      --environment "Variables={$ENV_STRING}" > /dev/null
    
    echo "✅ Environment variable updated."
  else
    echo "⚠️  Failed to update environment variables. Update BEDROCK_MODEL_ID manually in Lambda console."
  fi
else
  echo "⚠️  No environment variables found. Set environment variables manually in Lambda console."
  echo "   Required: AWS_REGION, OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX, FLASK_EXECUTE_ENDPOINT"
  echo "   Optional: BEDROCK_MODEL_ID, ENABLE_EXECUTION_TESTING, TRAINIUM_EXECUTION_TIMEOUT"
fi

echo ""
echo "✅ $FUNCTION_NAME deployed successfully."
echo ""
echo "📋 Configuration:"
echo "  Handler: lambda_handler.lambda_handler"
echo "  Timeout: 900 seconds (15 minutes) - recommended for PDF processing"
echo "  Memory: 2048 MB - recommended for PDF processing"
echo ""
echo "⚠️  Make sure to set these environment variables in Lambda console:"
echo "   - AWS_REGION (e.g., us-east-1)"
echo "   - OPENSEARCH_ENDPOINT (your OpenSearch domain endpoint)"
echo "   - OPENSEARCH_INDEX (e.g., research-papers-v2)"
echo "   - FLASK_EXECUTE_ENDPOINT (e.g., http://1.2.3.4:8000/execute)"
echo "   - BEDROCK_MODEL_ID (optional, defaults to Claude 3 Haiku)"
