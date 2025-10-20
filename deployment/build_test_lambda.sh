#!/bin/bash

# Build script for Code Test Lambda Function

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

echo "Packaging Code Test Lambda Function..."

# Create deployment directory
DEPLOY_DIR="test_lambda_deploy"
PACKAGE_NAME="test_lambda.zip"

# Clean up previous builds
rm -rf $DEPLOY_DIR
rm -f $PACKAGE_NAME

# Create deployment directory
mkdir -p $DEPLOY_DIR

# Copy the Lambda function
cp test_lambda/lambda_function.py $DEPLOY_DIR/

# Install dependencies
echo "Installing dependencies..."
pip install -r test_lambda/requirements.txt -t $DEPLOY_DIR/

# Create deployment package
echo "Creating deployment package..."
cd $DEPLOY_DIR
zip -r ../$PACKAGE_NAME . -x "*.pyc" "*/__pycache__/*"
cd ..

echo "Package created: $PACKAGE_NAME"
echo "Package size: $(du -h $PACKAGE_NAME | cut -f1)"

# Clean up
rm -rf $DEPLOY_DIR

echo "Lambda package ready for deployment!"
echo ""
echo "Next steps:"
echo "1. Upload $PACKAGE_NAME to AWS Lambda"
echo "2. Set handler to: lambda_function.lambda_handler"
echo "3. Set timeout to 900 seconds (15 minutes) to allow for Trainium communication"
echo "4. Set memory to 512 MB (lightweight dispatcher, no heavy computation)"
echo "5. Deploy Lambda in same VPC as Trainium instance"
echo "6. Configure SQS trigger:"
echo "   - Set BatchSize to 10"
echo "   - Set MaximumBatchingWindowInSeconds to 60"
echo "7. Configure environment variables:"
echo "   - OPENSEARCH_ENDPOINT"
echo "   - OPENSEARCH_INDEX (default: research-papers-v2)"
echo "   - OUTPUTS_BUCKET (default: papers-test-outputs)"
echo "   - TRAINIUM_ENDPOINT (e.g., http://10.0.1.50:8000)"
echo "   - TRAINIUM_INSTANCE_ID (optional, for auto-start)"
echo "   - BATCH_SIZE (default: 10)"
echo "   - TRAINIUM_TIMEOUT (default: 600)"
echo "8. Ensure IAM role has permissions for:"
echo "   - S3 read/write"
echo "   - SQS receive/delete messages"
echo "   - OpenSearch read/write"
echo "   - EC2 describe/start instances (if using auto-start)"

