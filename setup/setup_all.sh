#!/bin/bash
# Main setup script for Annapurna Pipeline
# This script runs all setup scripts in order

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Annapurna Pipeline - Complete Setup"
echo "=========================================="
echo ""

# Verify AWS account
echo "Verifying AWS account..."
if ! aws sts get-caller-identity &>/dev/null; then
    echo "Error: Unable to get AWS credentials."
    echo "Please configure AWS CLI: aws configure"
    exit 1
fi

ACCOUNT_INFO=$(aws sts get-caller-identity --output json)
ACCOUNT_ID=$(echo "$ACCOUNT_INFO" | jq -r '.Account')
USER_ARN=$(echo "$ACCOUNT_INFO" | jq -r '.Arn')

echo "Current AWS Account: $ACCOUNT_ID"
echo "Current User/Role: $USER_ARN"
echo ""

if [ -n "$AWS_PROFILE" ]; then
    echo "Using AWS Profile: $AWS_PROFILE"
    echo ""
fi

echo "This script will set up all AWS infrastructure for the Annapurna pipeline."
echo "Make sure you have:"
echo "  1. AWS CLI configured with appropriate credentials"
echo "  2. Required permissions to create IAM roles, Lambda functions, S3 buckets, etc."
echo "  3. jq installed (for JSON processing)"
echo ""
read -p "Continue with account $ACCOUNT_ID? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install it with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

# Set AWS region if not set
export AWS_REGION=${AWS_REGION:-us-east-1}
echo "Using AWS Region: $AWS_REGION"
echo ""

# Run setup scripts in order
echo "=========================================="
echo "Step 1: Setting up IAM Roles"
echo "=========================================="
bash 01_setup_iam_roles.sh

echo ""
echo "=========================================="
echo "Step 2: Setting up S3 Buckets"
echo "=========================================="
bash 02_setup_s3_buckets.sh

echo ""
echo "=========================================="
echo "Step 3: Setting up SQS Queues"
echo "=========================================="
bash 03_setup_sqs_queues.sh

echo ""
echo "=========================================="
echo "Step 4: Setting up Lambda Functions"
echo "=========================================="
echo "Note: This script configures Lambda settings but does not deploy code."
echo "You'll need to deploy Lambda function code separately using deployment scripts."
bash 04_setup_lambda_functions.sh

echo ""
echo "=========================================="
echo "Step 5: Setting up Step Functions"
echo "=========================================="
echo "Note: You'll need to update ARNs in the Step Function definition file."
bash 05_setup_step_functions.sh

echo ""
echo "=========================================="
echo "Step 6: Setting up DynamoDB"
echo "=========================================="
bash 06_setup_dynamodb.sh

echo ""
echo "=========================================="
echo "Step 7: OpenSearch Setup Instructions"
echo "=========================================="
bash 07_setup_opensearch.sh

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "  1. Review and update ARNs in conferenceScraper_definition.json"
echo "  2. Create OpenSearch domain manually (see 07_setup_opensearch.sh output)"
echo "  3. Deploy Lambda function code using deployment scripts"
echo "  4. Update Lambda environment variables with actual OpenSearch endpoint"
echo "  5. Upload dataset files to datasets-for-all-papers S3 bucket"
echo ""
echo "For detailed instructions, see setup/README.md"

