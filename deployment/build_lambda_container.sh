#!/bin/bash
# Build and deploy Lambda function as a container image

set -e

FUNCTION_NAME="PapersCodeGenerator-container"
REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
# Use the existing ECR repository (paperscodegenerator-lambda) that was created for the original function
ECR_REPO_NAME="paperscodegenerator-lambda"
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "🐳 Building Lambda container image..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# If script is in project root, use it; otherwise go up one level
if [ -d "$SCRIPT_DIR/code_gen" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

# Navigate to code-gen directory
if [ ! -d "code_gen" ]; then
    echo "❌ Error: code_gen not found"
    echo "   Current directory: $(pwd)"
    exit 1
fi
cd code_gen

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Create ECR repository if it doesn't exist
echo "📦 Creating ECR repository if needed..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${REGION} --image-scanning-configuration scanOnPush=true

# Ensure buildx is available and create a builder if needed
echo "🔧 Setting up Docker buildx..."
docker buildx version > /dev/null 2>&1 || (echo "❌ docker buildx not available" && exit 1)
docker buildx inspect lambda-builder > /dev/null 2>&1 || docker buildx create --name lambda-builder --use

# Build Docker image locally for linux/amd64, then push (avoids manifest list issue)
echo "🔨 Building Docker image locally..."
docker buildx build \
  --builder lambda-builder \
  --platform linux/amd64 \
  --load \
  -t ${ECR_REPO_NAME}:latest \
  .

# Tag for ECR
docker tag ${ECR_REPO_NAME}:latest ${ECR_REPO}:latest

# Push to ECR (regular docker push should avoid manifest list)
echo "📤 Pushing to ECR..."
docker push ${ECR_REPO}:latest

# Get the image digest to use for Lambda (avoids manifest list issues)
echo "🔍 Getting image digest..."
IMAGE_DIGEST=$(docker inspect ${ECR_REPO}:latest --format='{{index .RepoDigests 0}}' 2>/dev/null | cut -d'@' -f2)

# Update Lambda function to use container image
echo "🚀 Updating Lambda function..."
if [ -n "$IMAGE_DIGEST" ]; then
  echo "   Using image digest: ${ECR_REPO}@${IMAGE_DIGEST}"
  aws lambda update-function-code \
    --function-name ${FUNCTION_NAME} \
    --image-uri ${ECR_REPO}@${IMAGE_DIGEST} \
    --region ${REGION}
else
  echo "   Using image tag: ${ECR_REPO}:latest"
  aws lambda update-function-code \
    --function-name ${FUNCTION_NAME} \
    --image-uri ${ECR_REPO}:latest \
    --region ${REGION}
fi

echo ""
echo "✅ Lambda container deployed!"
echo "   Image: ${ECR_REPO}:latest"
echo ""
echo "⏳ Waiting for Lambda to be ready..."
sleep 10

echo "✅ Lambda should be ready. Test with:"
echo "   aws lambda invoke --function-name ${FUNCTION_NAME} --payload file://payload.json response.json"

