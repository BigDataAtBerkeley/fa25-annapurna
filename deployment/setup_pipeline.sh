#!/bin/bash

# Complete setup script for Code Generation & Testing Pipeline

set -e

echo "=========================================="
echo "Code Generation & Testing Pipeline Setup"
echo "=========================================="
echo ""

REGION="us-east-1"
CODE_BUCKET="papers-code-artifacts"
OUTPUTS_BUCKET="papers-test-outputs"
QUEUE_NAME="code-evaluation.fifo"

# Step 1: Create S3 buckets
echo "[Step 1/3] Creating S3 buckets..."
aws s3 mb s3://${CODE_BUCKET} --region ${REGION} 2>/dev/null || echo "Bucket ${CODE_BUCKET} already exists"
aws s3 mb s3://${OUTPUTS_BUCKET} --region ${REGION} 2>/dev/null || echo "Bucket ${OUTPUTS_BUCKET} already exists"
echo "✓ S3 buckets ready"

# Step 2: Get SQS queue URL (already exists)
echo ""
echo "[Step 2/3] Getting SQS queue URL..."
QUEUE_URL=$(aws sqs get-queue-url --queue-name ${QUEUE_NAME} --query 'QueueUrl' --output text 2>/dev/null)

if [ -z "$QUEUE_URL" ]; then
  echo "✗ SQS queue ${QUEUE_NAME} not found. Please create it first."
  exit 1
fi

echo "✓ SQS queue found: ${QUEUE_URL}"

# Step 3: Create IAM policies
echo ""
echo "[Step 3/3] Creating IAM policies..."

# Policy for CodeGenerator Lambda
cat > code-generator-policy.json << EOF
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
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:${REGION}::foundation-model/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost"
      ],
      "Resource": "arn:aws:es:${REGION}:*:domain/*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::${CODE_BUCKET}/*",
        "arn:aws:s3:::llm-research-papers/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sqs:SendMessage"
      ],
      "Resource": "arn:aws:sqs:${REGION}:*:${QUEUE_NAME}"
    }
  ]
}
EOF

# Policy for CodeTester Lambda
cat > code-tester-policy.json << EOF
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
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::${CODE_BUCKET}/*",
        "arn:aws:s3:::${OUTPUTS_BUCKET}/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:${REGION}:*:${QUEUE_NAME}"
    },
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost",
        "es:ESHttpPut"
      ],
      "Resource": "arn:aws:es:${REGION}:*:domain/*/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:StartInstances",
        "ec2:StopInstances"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "ec2:ResourceTag/Purpose": "PapersCodeTester"
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:CreateNetworkInterface",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DeleteNetworkInterface",
        "ec2:AssignPrivateIpAddresses",
        "ec2:UnassignPrivateIpAddresses"
      ],
      "Resource": "*"
    }
  ]
}
EOF

echo "✓ IAM policy files created"

# Step 4: Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Resources:"
echo "=========================================="
echo "S3 Buckets:"
echo "  - ${CODE_BUCKET} (for generated code)"
echo "  - ${OUTPUTS_BUCKET} (for test results)"
echo ""
echo "SQS Queue:"
echo "  - ${QUEUE_NAME}"
echo "  - URL: ${QUEUE_URL}"
echo ""
echo "Data Storage:"
echo "  - OpenSearch (papers + code metadata + test results)"
echo ""
echo "IAM Policies (files created, need to be applied):"
echo "  - code-generator-policy.json"
echo "  - code-tester-policy.json"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Deploy PapersCodeGenerator Lambda:"
echo "   ./build_code_gen_lambda.sh"
echo "   aws lambda create-function \\"
echo "     --function-name PapersCodeGenerator \\"
echo "     --runtime python3.9 \\"
echo "     --handler lambda_handler.lambda_handler \\"
echo "     --zip-file fileb://code_gen_lambda.zip \\"
echo "     --timeout 300 --memory-size 1024"
echo ""
echo "2. Deploy PapersCodeTester Lambda:"
echo "   ./build_test_lambda.sh"
echo "   aws lambda create-function \\"
echo "     --function-name PapersCodeTester \\"
echo "     --runtime python3.9 \\"
echo "     --handler lambda_function.lambda_handler \\"
echo "     --zip-file fileb://code_test_lambda.zip \\"
echo "     --timeout 900 --memory-size 512 \\"
echo "     --vpc-config SubnetIds=<SUBNET_ID>,SecurityGroupIds=<SG_ID>"
echo ""
echo "3. Configure environment variables:"
echo "   PapersCodeGenerator:"
echo "     CODE_BUCKET=${CODE_BUCKET}"
echo "     CODE_QUEUE_URL=${QUEUE_URL}"
echo "     OPENSEARCH_ENDPOINT=<your-opensearch-endpoint>"
echo "     OPENSEARCH_INDEX=research-papers-v2"
echo ""
echo "   PapersCodeTester:"
echo "     OUTPUTS_BUCKET=${OUTPUTS_BUCKET}"
echo "     OPENSEARCH_ENDPOINT=<your-opensearch-endpoint>"
echo "     OPENSEARCH_INDEX=research-papers-v2"
echo "     TRAINIUM_ENDPOINT=http://<trainium-private-ip>:8000"
echo "     TRAINIUM_INSTANCE_ID=<your-trainium-instance-id>"
echo "     BATCH_SIZE=10"
echo "     TRAINIUM_TIMEOUT=600"
echo ""
echo "4. Setup Trainium Instance:"
echo "   - Launch trn1.2xlarge instance in same VPC"
echo "   - Install PyTorch Neuron and dependencies"
echo "   - Deploy trainium executor service (see README)"
echo "   - Tag instance with Purpose=PapersCodeTester"
echo "   - Configure security group to allow Lambda access on port 8000"
echo ""
echo "5. Attach SQS trigger to PapersCodeTester:"
echo "   - BatchSize: 10"
echo "   - MaximumBatchingWindowInSeconds: 60"
echo ""
echo "6. Test the pipeline:"
echo "   aws lambda invoke \\"
echo "     --function-name PapersCodeGenerator \\"
echo "     --payload '{\"action\":\"generate_by_title\",\"title\":\"ResNet\"}' \\"
echo "     response.json"
echo ""
echo "=========================================="

# Clean up policy files
rm code-generator-policy.json code-tester-policy.json

