#!/bin/bash
# Setup IAM Roles for Annapurna Pipeline

set -e

echo "=== Setting up IAM Roles ==="

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}

echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"

# 1. Create PaperScraper Role
echo "Creating PaperScraper IAM Role..."

# Trust policy for Lambda
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)

# Create role if it doesn't exist
if aws iam get-role --role-name PaperScraper 2>/dev/null; then
    echo "Role PaperScraper already exists, skipping creation..."
else
    aws iam create-role \
        --role-name PaperScraper \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "IAM role for Lambda scraper functions"
    echo "Created PaperScraper role"
fi

# Attach policies
echo "Attaching policies to PaperScraper role..."

# Basic Lambda execution
aws iam attach-role-policy \
    --role-name PaperScraper \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# S3 Full Access
aws iam attach-role-policy \
    --role-name PaperScraper \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# SQS Full Access
aws iam attach-role-policy \
    --role-name PaperScraper \
    --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess

echo "✓ PaperScraper role configured"

# 2. Create PapersJudge Role (will be created automatically by Lambda, but we'll ensure it exists)
echo "Creating PapersJudge IAM Role..."

if aws iam get-role --role-name PapersJudge-role 2>/dev/null; then
    echo "Role PapersJudge-role already exists, skipping creation..."
else
    aws iam create-role \
        --role-name PapersJudge-role \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "IAM role for PapersJudge Lambda function"
    echo "Created PapersJudge-role"
fi

# Attach Bedrock Full Access
aws iam attach-role-policy \
    --role-name PapersJudge-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
    --role-name PapersJudge-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

# OpenSearch access (inline policy)
OPENSEARCH_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost",
        "es:ESHttpPut",
        "es:DescribeElasticsearchDomain",
        "es:DescribeElasticsearchDomains"
      ],
      "Resource": "*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name PapersJudge-role \
    --policy-name OpenSearchAccess \
    --policy-document "$OPENSEARCH_POLICY"

# S3 access for discarded papers
aws iam attach-role-policy \
    --role-name PapersJudge-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# SQS access
aws iam attach-role-policy \
    --role-name PapersJudge-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess

echo "✓ PapersJudge-role configured"

# 3. Create ECS Task Role (for PapersCodeGenerator-container)
echo "Creating ECS Task Role..."

if aws iam get-role --role-name annapurna-ecs-task-role 2>/dev/null; then
    echo "Role annapurna-ecs-task-role already exists, skipping creation..."
else
    ECS_TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)
    aws iam create-role \
        --role-name annapurna-ecs-task-role \
        --assume-role-policy-document "$ECS_TRUST_POLICY" \
        --description "ECS task role for Annapurna containers"
    echo "Created annapurna-ecs-task-role"
fi

# Create AnnapurnaTaskPolicy
TASK_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::papers-code-artifacts",
        "arn:aws:s3:::papers-code-artifacts/*",
        "arn:aws:s3:::llm-research-papers",
        "arn:aws:s3:::llm-research-papers/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sqs:SendMessage",
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:ChangeMessageVisibility"
      ],
      "Resource": "arn:aws:sqs:${REGION}:${ACCOUNT_ID}:trainium-execution.fifo"
    },
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost",
        "es:ESHttpPut"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name annapurna-ecs-task-role \
    --policy-name AnnapurnaTaskPolicy \
    --policy-document "$TASK_POLICY"

echo "✓ annapurna-ecs-task-role configured"

# 4. Create ECS Execution Role
echo "Creating ECS Execution Role..."

if aws iam get-role --role-name annapurna-ecs-execution-role 2>/dev/null; then
    echo "Role annapurna-ecs-execution-role already exists, skipping creation..."
else
    aws iam create-role \
        --role-name annapurna-ecs-execution-role \
        --assume-role-policy-document "$ECS_TRUST_POLICY" \
        --description "ECS execution role for Annapurna containers"
    echo "Created annapurna-ecs-execution-role"
fi

# Attach ECS execution policy
aws iam attach-role-policy \
    --role-name annapurna-ecs-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create AnnapurnaExecutionPolicy
EXECUTION_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name annapurna-ecs-execution-role \
    --policy-name AnnapurnaExecutionPolicy \
    --policy-document "$EXECUTION_POLICY"

echo "✓ annapurna-ecs-execution-role configured"

# 5. Create PapersCodeGenerator-container Role
echo "Creating PapersCodeGenerator-container IAM Role..."

if aws iam get-role --role-name PapersCodeGenerator-container-role 2>/dev/null; then
    echo "Role PapersCodeGenerator-container-role already exists, skipping creation..."
else
    aws iam create-role \
        --role-name PapersCodeGenerator-container-role \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "IAM role for PapersCodeGenerator-container Lambda function"
    echo "Created PapersCodeGenerator-container-role"
fi

# Attach policies
aws iam attach-role-policy \
    --role-name PapersCodeGenerator-container-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
    --role-name PapersCodeGenerator-container-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

aws iam attach-role-policy \
    --role-name PapersCodeGenerator-container-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name PapersCodeGenerator-container-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess

# OpenSearch access (inline policy)
OPENSEARCH_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost",
        "es:ESHttpPut",
        "es:DescribeElasticsearchDomain",
        "es:DescribeElasticsearchDomains"
      ],
      "Resource": "*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name PapersCodeGenerator-container-role \
    --policy-name OpenSearchAccess \
    --policy-document "$OPENSEARCH_POLICY"

echo "✓ PapersCodeGenerator-container-role configured"

echo ""
echo "=== IAM Roles Setup Complete ==="

