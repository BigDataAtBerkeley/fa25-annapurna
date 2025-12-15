#!/bin/bash
# Setup DynamoDB Table for APL Pipeline

set -e

echo "=== Setting up DynamoDB Table ==="

REGION=${AWS_REGION:-us-east-2}
TABLE_NAME="docRunErrors"

echo "Region: $REGION"
echo "Table: $TABLE_NAME"

# Check if table exists
if aws dynamodb describe-table --table-name "$TABLE_NAME" --region "$REGION" 2>/dev/null; then
    echo "  Table $TABLE_NAME already exists, skipping creation..."
else
    echo "  Creating DynamoDB table: $TABLE_NAME"
    
    aws dynamodb create-table \
        --table-name "$TABLE_NAME" \
        --attribute-definitions \
            AttributeName=docID,AttributeType=S \
            AttributeName=interationNum,AttributeType=S \
        --key-schema \
            AttributeName=docID,KeyType=HASH \
            AttributeName=interationNum,KeyType=RANGE \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION" > /dev/null
    
    echo "  Waiting for table to be active..."
    aws dynamodb wait table-exists \
        --table-name "$TABLE_NAME" \
        --region "$REGION"
    
    echo "  ✓ Created table: $TABLE_NAME"
fi

echo ""
echo "=== DynamoDB Setup Complete ==="

