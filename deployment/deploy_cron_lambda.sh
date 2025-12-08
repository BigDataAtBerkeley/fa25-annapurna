#!/bin/bash
#
# Deploy Cron Lambda function with IAM role and environment variables
#

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

AWS_REGION="${AWS_REGION:-us-east-1}"
FUNCTION_NAME="${FUNCTION_NAME:-PapersCronJob}"
ROLE_NAME="${ROLE_NAME:-PapersCronJobRole}"
POLICY_NAME="${POLICY_NAME:-PapersCronJobPolicy}"
ZIP_FILE="${ZIP_FILE:-$PROJECT_ROOT/cron_lambda/build/cron_lambda.zip}"

echo "========================================="
echo "Deploying Cron Lambda Function"
echo "========================================="
echo "Region: $AWS_REGION"
echo "Function: $FUNCTION_NAME"
echo "Role: $ROLE_NAME"
echo ""

# Get account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account ID: $ACCOUNT_ID"
echo ""

# Step 1: Check if Lambda package exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ Lambda package not found: $ZIP_FILE"
    echo "   Please run ./deployment/build_cron_lambda.sh first"
    exit 1
fi

PACKAGE_SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null || ls -l "$ZIP_FILE" | awk '{print $5}')
echo "✅ Lambda package found: $ZIP_FILE (${PACKAGE_SIZE} bytes)"
echo ""

# Step 2: Create IAM role if it doesn't exist
echo "🔐 Setting up IAM role..."

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ]; then
    echo "📝 Creating IAM role: $ROLE_NAME"
    
    # Create trust policy for Lambda
    cat > /tmp/trust-policy.json << EOF
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

    ROLE_ARN=$(aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        --description "IAM role for PapersCronJob Lambda function" \
        --query 'Role.Arn' \
        --output text)
    
    echo "✅ Created IAM role: $ROLE_ARN"
    rm -f /tmp/trust-policy.json
else
    echo "✅ IAM role already exists: $ROLE_ARN"
fi

# Step 3: Create and attach IAM policy
echo ""
echo "📋 Creating IAM policy..."

# Get environment variables for policy (if set)
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-*}"
CODE_BUCKET="${CODE_BUCKET:-papers-code-artifacts}"
CODE_EVAL_QUEUE_NAME="${CODE_EVAL_QUEUE_NAME:-code-evaluation.fifo}"

# Extract OpenSearch domain from endpoint if provided
OPENSEARCH_DOMAIN="*"
if [[ "$OPENSEARCH_ENDPOINT" != "*" ]]; then
    # Extract domain name from endpoint (e.g., search-research-papers-xxx.us-east-1.es.amazonaws.com)
    OPENSEARCH_DOMAIN=$(echo "$OPENSEARCH_ENDPOINT" | sed 's|https\?://||' | sed 's|/.*||' | sed 's|\..*||' | sed 's|^search-||')
fi

cat > /tmp/lambda-policy.json << EOF
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
      "Resource": "arn:aws:logs:${AWS_REGION}:${ACCOUNT_ID}:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost",
        "es:ESHttpPut"
      ],
      "Resource": "arn:aws:es:${AWS_REGION}:${ACCOUNT_ID}:domain/${OPENSEARCH_DOMAIN}/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sqs:SendMessage",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:${AWS_REGION}:${ACCOUNT_ID}:${CODE_EVAL_QUEUE_NAME}"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:HeadObject"
      ],
      "Resource": "arn:aws:s3:::${CODE_BUCKET}/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricStatistics"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "lambda:GetFunctionConfiguration"
      ],
      "Resource": "arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:StartInstances",
        "ec2:StopInstances"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Check if policy exists
POLICY_ARN=$(aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}" --query 'Policy.Arn' --output text 2>/dev/null || echo "")

if [ -z "$POLICY_ARN" ]; then
    echo "📝 Creating IAM policy: $POLICY_NAME"
    POLICY_ARN=$(aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document file:///tmp/lambda-policy.json \
        --description "Policy for PapersCronJob Lambda function" \
        --query 'Policy.Arn' \
        --output text)
    echo "✅ Created IAM policy: $POLICY_ARN"
else
    echo "📝 Updating existing IAM policy: $POLICY_NAME"
    
    # Check how many policy versions exist
    VERSION_COUNT=$(aws iam list-policy-versions \
        --policy-arn "$POLICY_ARN" \
        --query 'length(Versions)' \
        --output text 2>/dev/null || echo "0")
    
    # If we have 5 versions (the limit), delete the oldest non-default version
    if [ "$VERSION_COUNT" -ge 5 ]; then
        echo "⚠️  Policy has 5 versions (limit reached). Deleting oldest non-default version..."
        
        # Get the default version ID
        DEFAULT_VERSION=$(aws iam get-policy \
            --policy-arn "$POLICY_ARN" \
            --query 'Policy.DefaultVersionId' \
            --output text 2>/dev/null)
        
        # Get the oldest non-default version (sorted by CreateDate, excluding default)
        OLDEST_VERSION=$(aws iam list-policy-versions \
            --policy-arn "$POLICY_ARN" \
            --query "Versions[?VersionId != \`$DEFAULT_VERSION\`] | sort_by(@, &CreateDate) | [0].VersionId" \
            --output text 2>/dev/null)
        
        DELETED=false
        if [ -n "$OLDEST_VERSION" ] && [ "$OLDEST_VERSION" != "None" ] && [ "$OLDEST_VERSION" != "null" ]; then
            echo "🗑️  Attempting to delete oldest non-default policy version: $OLDEST_VERSION"
            if aws iam delete-policy-version \
                --policy-arn "$POLICY_ARN" \
                --version-id "$OLDEST_VERSION" 2>&1; then
                echo "✅ Deleted policy version: $OLDEST_VERSION"
                DELETED=true
            else
                echo "❌ Failed to delete policy version: $OLDEST_VERSION"
            fi
        fi
        
        if [ "$DELETED" = false ]; then
            echo "⚠️  Warning: Could not delete any non-default policy version. You may need to manually delete a version."
            echo "   Run: aws iam list-policy-versions --policy-arn $POLICY_ARN"
            echo "   Then: aws iam delete-policy-version --policy-arn $POLICY_ARN --version-id <VERSION_ID>"
        fi
    fi
    
    # Now create the new policy version
    aws iam create-policy-version \
        --policy-arn "$POLICY_ARN" \
        --policy-document file:///tmp/lambda-policy.json \
        --set-as-default > /dev/null
    echo "✅ Updated IAM policy"
fi

# Attach policy to role
echo "🔗 Attaching policy to role..."
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "$POLICY_ARN" 2>/dev/null || echo "  (Policy already attached)"

rm -f /tmp/lambda-policy.json

echo "✅ IAM setup complete"
echo ""

# Step 4: Wait for role to be ready (IAM propagation delay)
echo "⏳ Waiting for IAM role to propagate (5 seconds)..."
sleep 5

# Step 5: Create or update Lambda function
echo ""
echo "🚀 Deploying Lambda function..."

# Check if function exists
FUNCTION_EXISTS=$(aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" 2>/dev/null || echo "")

if [ -z "$FUNCTION_EXISTS" ]; then
    echo "📝 Creating new Lambda function: $FUNCTION_NAME"
    
    # Check if package is too large for direct upload (>50MB)
    if [ $PACKAGE_SIZE -gt 52428800 ]; then
        echo "⚠️  Package is larger than 50MB, uploading to S3 first..."
        
        S3_BUCKET="${LAMBDA_DEPLOY_BUCKET:-papers-code-artifacts}"
        S3_KEY="lambda-deployments/${FUNCTION_NAME}-$(date +%s).zip"
        
        echo "📤 Uploading to s3://${S3_BUCKET}/${S3_KEY}..."
        aws s3 cp "$ZIP_FILE" "s3://${S3_BUCKET}/${S3_KEY}"
        
        aws lambda create-function \
            --function-name "$FUNCTION_NAME" \
            --runtime python3.11 \
            --role "$ROLE_ARN" \
            --handler lambda_function.lambda_handler \
            --s3-bucket "${S3_BUCKET}" \
            --s3-key "${S3_KEY}" \
            --timeout 300 \
            --memory-size 512 \
            --region "$AWS_REGION" \
            --description "Cron job Lambda that processes papers through the pipeline every 30 minutes"
    else
        aws lambda create-function \
            --function-name "$FUNCTION_NAME" \
            --runtime python3.11 \
            --role "$ROLE_ARN" \
            --handler lambda_function.lambda_handler \
            --zip-file "fileb://$ZIP_FILE" \
            --timeout 300 \
            --memory-size 512 \
            --region "$AWS_REGION" \
            --description "Cron job Lambda that processes papers through the pipeline every 30 minutes"
    fi
    
    echo "✅ Lambda function created"
else
    echo "📝 Updating existing Lambda function: $FUNCTION_NAME"
    
    # Check if package is too large for direct upload (>50MB)
    if [ $PACKAGE_SIZE -gt 52428800 ]; then
        echo "⚠️  Package is larger than 50MB, uploading to S3 first..."
        
        S3_BUCKET="${LAMBDA_DEPLOY_BUCKET:-papers-code-artifacts}"
        S3_KEY="lambda-deployments/${FUNCTION_NAME}-$(date +%s).zip"
        
        echo "📤 Uploading to s3://${S3_BUCKET}/${S3_KEY}..."
        aws s3 cp "$ZIP_FILE" "s3://${S3_BUCKET}/${S3_KEY}"
        
        aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --s3-bucket "${S3_BUCKET}" \
            --s3-key "${S3_KEY}" \
            --region "$AWS_REGION"
    else
        aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --zip-file "fileb://$ZIP_FILE" \
            --region "$AWS_REGION"
    fi
    
    echo "✅ Lambda function code updated"
fi

# Step 6: Set environment variables
echo ""
echo "🔧 Setting environment variables..."

# Load .env file if it exists (from project root)
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    echo "📄 Loading environment variables from .env file..."
    # Read .env file and export variables (skip comments and empty lines)
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        # Export the variable (handles KEY=value format, strips quotes)
        if [[ "$line" =~ ^[[:space:]]*([^=]+)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]// /}"
            value="${BASH_REMATCH[2]}"
            # Remove surrounding quotes if present
            if [[ "$value" =~ ^\"(.*)\"$ ]] || [[ "$value" =~ ^\'(.*)\'$ ]]; then
                value="${BASH_REMATCH[1]}"
            fi
            export "$key"="$value"
        fi
    done < "$ENV_FILE"
    echo "✅ Loaded variables from .env file"
fi

if [ -z "$OPENSEARCH_ENDPOINT" ] || [ "$OPENSEARCH_ENDPOINT" == "*" ]; then
    read -p "Enter OpenSearch endpoint (e.g., https://search-xxx.us-east-1.es.amazonaws.com): " OPENSEARCH_ENDPOINT
fi

if [ -z "$OPENSEARCH_INDEX" ]; then
    read -p "Enter OpenSearch index (default: research-papers-v3): " OPENSEARCH_INDEX
    OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-research-papers-v3}"
fi

if [ -z "$CODE_EVAL_QUEUE_URL" ]; then
    read -p "Enter code-evaluation queue URL (or press Enter to auto-detect): " CODE_EVAL_QUEUE_URL
    if [ -z "$CODE_EVAL_QUEUE_URL" ]; then
        CODE_EVAL_QUEUE_URL=$(aws sqs get-queue-url \
            --queue-name "code-evaluation.fifo" \
            --region "$AWS_REGION" \
            --query 'QueueUrl' \
            --output text 2>/dev/null || echo "")
        if [ -z "$CODE_EVAL_QUEUE_URL" ]; then
            echo "⚠️  Could not auto-detect queue URL. Please set CODE_EVAL_QUEUE_URL manually."
        fi
    fi
fi

if [ -z "$TRAINIUM_ENDPOINT" ]; then
    read -p "Enter Trainium endpoint (e.g., http://1.2.3.4:8000) or press Enter to skip: " TRAINIUM_ENDPOINT
fi

if [ -z "$TRAINIUM_INSTANCE_ID" ]; then
    read -p "Enter Trainium EC2 instance ID (e.g., i-0f0bf0de25aa4fd57) or press Enter to skip auto start/stop: " TRAINIUM_INSTANCE_ID
fi

# Build environment variables in AWS CLI format (Key=Value pairs)
# Use a temporary JSON file to avoid shell escaping issues
TEMP_ENV_FILE=$(mktemp)
cat > "$TEMP_ENV_FILE" << EOF
{
  "Variables": {
    "OPENSEARCH_ENDPOINT": "${OPENSEARCH_ENDPOINT}",
    "OPENSEARCH_INDEX": "${OPENSEARCH_INDEX}",
    "CODE_EVAL_QUEUE_URL": "${CODE_EVAL_QUEUE_URL}",
    "CODE_BUCKET": "${CODE_BUCKET}",
    "MAX_CODE_GEN_CONCURRENT": "5",
    "MAX_TRAINIUM_CONCURRENT": "1",
    "BATCH_SIZE_FOR_EXECUTION": "10",
    "MAX_PAPERS_PER_RUN": "5",
    "CODE_GEN_LAMBDA_NAME": "PapersCodeGenerator-container",
    "TRAINIUM_EXECUTION_TIMEOUT": "3600"
EOF

if [ -n "$TRAINIUM_ENDPOINT" ]; then
    cat >> "$TEMP_ENV_FILE" << EOF
    ,"TRAINIUM_ENDPOINT": "${TRAINIUM_ENDPOINT}"
EOF
fi

if [ -n "$TRAINIUM_INSTANCE_ID" ]; then
    cat >> "$TEMP_ENV_FILE" << EOF
    ,"TRAINIUM_INSTANCE_ID": "${TRAINIUM_INSTANCE_ID}"
EOF
fi

cat >> "$TEMP_ENV_FILE" << EOF
  }
}
EOF

echo "📝 Updating Lambda environment variables..."
aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --environment file://"$TEMP_ENV_FILE" \
    --region "$AWS_REGION" > /dev/null

rm -f "$TEMP_ENV_FILE"

echo "✅ Environment variables set"
echo ""

# Step 7: Summary
echo "========================================="
echo "✅ Deployment Complete!"
echo "========================================="
echo ""
echo "📋 Lambda Configuration:"
echo "  Function: $FUNCTION_NAME"
echo "  Role: $ROLE_ARN"
echo "  Handler: lambda_function.lambda_handler"
echo "  Timeout: 300 seconds"
echo "  Memory: 512 MB"
echo ""
echo "📋 Environment Variables:"
echo "  AWS_REGION: $AWS_REGION (automatically provided by Lambda)"
echo "  OPENSEARCH_ENDPOINT: $OPENSEARCH_ENDPOINT"
echo "  OPENSEARCH_INDEX: $OPENSEARCH_INDEX"
echo "  CODE_EVAL_QUEUE_URL: $CODE_EVAL_QUEUE_URL"
echo "  CODE_BUCKET: $CODE_BUCKET"
if [ -n "$TRAINIUM_ENDPOINT" ]; then
    echo "  TRAINIUM_ENDPOINT: $TRAINIUM_ENDPOINT"
fi
echo ""
echo "📋 Next Steps:"
echo "  1. Setup EventBridge rule (runs every 30 minutes):"
echo "     ./deployment/setup_cron_job.sh"
echo ""
echo "  2. Test the Lambda:"
echo "     aws lambda invoke \\"
echo "       --function-name $FUNCTION_NAME \\"
echo "       --region $AWS_REGION \\"
echo "       response.json"
echo ""
echo "  3. View logs:"
echo "     aws logs tail /aws/lambda/$FUNCTION_NAME --follow --region $AWS_REGION"
echo ""

