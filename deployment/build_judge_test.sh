#!/bin/bash
set -e

# === Configuration ===
FUNCTION_NAME="PaperJudgeTest"          # Lambda function name
ZIP_FILE="judge_test.zip"                # Output zip file name
JUDGE_DIR="../judge_lambda_test"         # Directory containing Lambda code
DEPLOY_DIR="../deploy"                   # Deployment output directory
REQ_FILE="$DEPLOY_DIR/requirements_judge.txt"

echo "📦 Packaging and deploying $FUNCTION_NAME..."

# === Step 1: Clean old zip ===
echo "🧹 Cleaning old build files..."
rm -f $DEPLOY_DIR/$ZIP_FILE
rm -f $ZIP_FILE

# === Step 2: Move into judge Lambda directory ===
cd $JUDGE_DIR

# === Step 3: Install dependencies locally ===
echo "📥 Installing dependencies..."
pip install -r $REQ_FILE -t .

# === Step 4: Zip everything (excluding cache & logs) ===
echo "🗜️ Creating deployment package..."
zip -r9 $DEPLOY_DIR/$ZIP_FILE . -x "*.log" "logs/*" "__pycache__/*" "*.pyc" "*.zip"

# === Step 5: Cleanup build artifacts ===
echo "🧹 Cleaning up temporary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + >/dev/null 2>&1
rm -rf boto3* opensearch_py*

# === Step 6: Deploy to AWS Lambda ===
cd $DEPLOY_DIR
echo "🚀 Deploying $FUNCTION_NAME to AWS Lambda..."
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb://$ZIP_FILE

echo "✅ $FUNCTION_NAME deployed successfully."
