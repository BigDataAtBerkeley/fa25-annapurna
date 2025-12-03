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

# Install dependencies for Linux (Lambda runs on Amazon Linux)
echo "📥 Installing dependencies for Linux (Lambda environment)..."
echo "Note: Installing all dependencies including pymupdf (using pre-built x86_64 wheel)"

# Download pre-built x86_64 wheel for pymupdf (more reliable than building from source)
echo "Downloading pre-built pymupdf wheel for x86_64..."
pip download --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.9 --no-deps pymupdf==1.23.0 -d /tmp/lambda-wheels 2>/dev/null || true

# Install all dependencies
echo "Installing dependencies..."
if [ -f /tmp/lambda-wheels/*.whl ]; then
    echo "Installing pymupdf from pre-built wheel..."
    pip install --find-links /tmp/lambda-wheels --no-index pymupdf==1.23.0 -t $DEPLOY_DIR/ || \
    pip install --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.9 pymupdf==1.23.0 -t $DEPLOY_DIR/
    rm -rf /tmp/lambda-wheels
fi

# Install other dependencies
pip install -r midpoint-deliverable/code-gen-for-deliv/requirements.txt -t $DEPLOY_DIR/ --no-deps || true
pip install boto3 opensearch-py python-dotenv requests "urllib3>=1.26.0,<2.0.0" Pillow -t $DEPLOY_DIR/

echo "✅ Dependencies installed"

# Create deployment package
echo "🗜️ Creating deployment package..."
cd $DEPLOY_DIR
zip -r9 ../deployment/$ZIP_FILE . -x "*.pyc" "*/__pycache__/*" "*/tests/*" "*/test_*"
cd ..

# Clean up
echo "🧹 Cleaning up temporary files..."
rm -rf "$DEPLOY_DIR" 2>/dev/null || true

echo "Package created: deployment/$ZIP_FILE"
# Get file size (works on both Linux and macOS)
PACKAGE_SIZE=$(stat -f%z "deployment/$ZIP_FILE" 2>/dev/null || stat -c%s "deployment/$ZIP_FILE" 2>/dev/null || ls -l "deployment/$ZIP_FILE" | awk '{print $5}')
PACKAGE_SIZE_MB=$((PACKAGE_SIZE / 1024 / 1024))
echo "Package size: ${PACKAGE_SIZE_MB}MB (${PACKAGE_SIZE} bytes)"

cd deployment

# Check if package is too large for direct upload (>50MB)
if [ $PACKAGE_SIZE -gt 52428800 ]; then
    echo "⚠️  Package is larger than 50MB, uploading to S3 first..."
    
    S3_BUCKET="${LAMBDA_DEPLOY_BUCKET:-papers-code-artifacts}"
    S3_KEY="lambda-deployments/${FUNCTION_NAME}-$(date +%s).zip"
    
    echo "📤 Uploading to s3://${S3_BUCKET}/${S3_KEY}..."
    aws s3 cp $ZIP_FILE s3://${S3_BUCKET}/${S3_KEY}
    
    echo "🚀 Updating $FUNCTION_NAME from S3..."
    aws lambda update-function-code \
      --function-name $FUNCTION_NAME \
      --s3-bucket ${S3_BUCKET} \
      --s3-key ${S3_KEY}
else
    echo "🚀 Updating $FUNCTION_NAME in AWS (direct upload)..."
    aws lambda update-function-code \
      --function-name $FUNCTION_NAME \
      --zip-file fileb://$ZIP_FILE
fi

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
