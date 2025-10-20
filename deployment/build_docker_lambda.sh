#!/bin/bash
# Build and deploy Docker-based Lambda for PapersCodeTester

set -e

echo "Building Docker Lambda for PapersCodeTester"
echo "================================================"

# Configuration
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-east-1"
ECR_REPO="papers-code-tester"
IMAGE_TAG="latest"
LAMBDA_FUNCTION="PapersCodeTester"

echo "Account ID: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "ECR Repo: $ECR_REPO"
echo ""

# Step 1: Create ECR repository if it doesn't exist
echo "[Step 1/5] Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION 2>/dev/null || \
  aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
echo "ECR repository ready"

# Step 2: Login to ECR
echo ""
echo "[Step 2/5] Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
echo "Logged into ECR"

# Step 3: Build Docker image
echo ""
echo "[Step 3/5] Building Docker image (this takes ~5-10 minutes)..."
docker build --platform linux/amd64 -t $ECR_REPO:$IMAGE_TAG -f test_lambda/Dockerfile .
echo "Docker image built"

# Step 4: Tag and push to ECR
echo ""
echo "[Step 4/5] Pushing to ECR..."
docker tag $ECR_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
echo "Image pushed to ECR"

# Step 5: Update Lambda function
echo ""
echo "[Step 5/5] Updating Lambda function..."
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

# Check if function exists
if aws lambda get-function --function-name $LAMBDA_FUNCTION 2>/dev/null; then
  # Update existing function
  aws lambda update-function-code \
    --function-name $LAMBDA_FUNCTION \
    --image-uri $IMAGE_URI \
    --region $AWS_REGION
  echo "Lambda function updated"
else
  echo "Lambda function doesn't exist. Creating it..."
  # You'll need to set your IAM role ARN
  echo "Run this command manually:"
  echo "aws lambda create-function \\"
  echo "  --function-name $LAMBDA_FUNCTION \\"
  echo "  --package-type Image \\"
  echo "  --code ImageUri=$IMAGE_URI \\"
  echo "  --role arn:aws:iam::$AWS_ACCOUNT_ID:role/YOUR_LAMBDA_ROLE \\"
  echo "  --timeout 900 \\"
  echo "  --memory-size 3008 \\"
  echo "  --environment Variables='{OPENSEARCH_ENDPOINT=your-endpoint,OPENSEARCH_INDEX=research-papers-v2,OUTPUTS_BUCKET=papers-test-outputs}'"
fi

echo ""
echo "================================================"
echo "Deployment complete!"
echo "================================================"
echo "Image URI: $IMAGE_URI"
echo ""
echo "Next steps:"
echo "1. If this is first time, create Lambda function using command above"
echo "2. Test the function: aws lambda invoke --function-name $LAMBDA_FUNCTION response.json"

