#!/bin/bash
# Setup S3 Buckets for Annapurna Pipeline

set -e

echo "=== Setting up S3 Buckets ==="

REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Make bucket names unique by appending account ID
# S3 bucket names are globally unique, so we need to ensure uniqueness
BUCKET_SUFFIX="${ACCOUNT_ID: -8}"  # Use last 8 digits of account ID

# List of buckets to create (with account-specific suffix)
BUCKETS=(
    "llm-research-papers-${BUCKET_SUFFIX}"
    "datasets-for-all-papers-${BUCKET_SUFFIX}"
    "discarded-papers-${BUCKET_SUFFIX}"
    "papers-code-artifacts-${BUCKET_SUFFIX}"
    "papers-test-outputs-${BUCKET_SUFFIX}"
    "trainium-execution-results-${BUCKET_SUFFIX}"
)

echo "Using account-specific bucket names (suffix: ${BUCKET_SUFFIX})"
echo ""

for BUCKET_NAME in "${BUCKETS[@]}"; do
    echo "Creating bucket: $BUCKET_NAME"
    
    # Check if bucket already exists in this account
    if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
        echo "  Bucket $BUCKET_NAME already exists in this account, skipping..."
    else
        # Create bucket
        if [ "$REGION" = "us-east-1" ]; then
            # us-east-1 doesn't need LocationConstraint
            aws s3api create-bucket \
                --bucket "$BUCKET_NAME" \
                --region "$REGION" 2>&1 | grep -v "BucketAlreadyOwnedByYou" || true
        else
            aws s3api create-bucket \
                --bucket "$BUCKET_NAME" \
                --region "$REGION" \
                --create-bucket-configuration LocationConstraint="$REGION" 2>&1 | grep -v "BucketAlreadyOwnedByYou" || true
        fi
        
        # Verify bucket was created
        if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
            echo "  ✓ Created bucket: $BUCKET_NAME"
        else
            echo "  ⚠ Bucket creation may have failed or bucket already exists globally"
        fi
    fi
    
    # Enable versioning (optional but recommended)
    aws s3api put-bucket-versioning \
        --bucket "$BUCKET_NAME" \
        --versioning-configuration Status=Enabled 2>/dev/null || true
    
    # Block public access
    aws s3api put-public-access-block \
        --bucket "$BUCKET_NAME" \
        --public-access-block-configuration \
        "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true" 2>/dev/null || true
done

# Create folder structure for datasets bucket
DATASETS_BUCKET="datasets-for-all-papers-${BUCKET_SUFFIX}"
echo "Creating folder structure in ${DATASETS_BUCKET}..."
aws s3api put-object --bucket "$DATASETS_BUCKET" --key cifar10/ 2>/dev/null || true
aws s3api put-object --bucket "$DATASETS_BUCKET" --key cifar100/ 2>/dev/null || true
aws s3api put-object --bucket "$DATASETS_BUCKET" --key fashion_mnist/ 2>/dev/null || true
aws s3api put-object --bucket "$DATASETS_BUCKET" --key imdb/ 2>/dev/null || true
aws s3api put-object --bucket "$DATASETS_BUCKET" --key mnist/ 2>/dev/null || true
aws s3api put-object --bucket "$DATASETS_BUCKET" --key synthetic/ 2>/dev/null || true
aws s3api put-object --bucket "$DATASETS_BUCKET" --key wikitext2/ 2>/dev/null || true

# Create rejected folder in discarded-papers bucket
DISCARDED_BUCKET="discarded-papers-${BUCKET_SUFFIX}"
echo "Creating rejected folder in ${DISCARDED_BUCKET}..."
aws s3api put-object --bucket "$DISCARDED_BUCKET" --key rejected/ 2>/dev/null || true

echo ""
echo "=== S3 Buckets Setup Complete ==="
echo ""
echo "Created buckets with account-specific names:"
for BUCKET_NAME in "${BUCKETS[@]}"; do
    echo "  - $BUCKET_NAME"
done
echo ""
echo "Note: Update Lambda environment variables to use these bucket names:"
echo "  BUCKET_NAME=${BUCKETS[0]}"
echo "  CODE_BUCKET=${BUCKETS[3]}"
echo ""
echo "You will need to upload dataset files to ${DATASETS_BUCKET} manually."
echo "See the README for the expected folder structure."

