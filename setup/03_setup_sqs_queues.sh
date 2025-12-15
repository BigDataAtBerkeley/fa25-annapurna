#!/bin/bash
# Setup SQS Queues for Annapurna Pipeline

set -e

echo "=== Setting up SQS Queues ==="

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}

echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"

# Function to get role ARN
get_role_arn() {
    local role_name=$1
    aws iam get-role --role-name "$role_name" --query 'Role.Arn' --output text 2>/dev/null || echo ""
}

PAPER_SCRAPER_ROLE_ARN=$(get_role_arn PaperScraper)
PAPERS_JUDGE_ROLE_ARN=$(get_role_arn PapersJudge-role)

if [ -z "$PAPER_SCRAPER_ROLE_ARN" ]; then
    echo "Warning: PaperScraper role not found. Please run 01_setup_iam_roles.sh first."
    exit 1
fi

# 1. researchQueue.fifo
echo "Creating researchQueue.fifo..."
QUEUE_NAME="researchQueue.fifo"
QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text 2>/dev/null || echo "")

if [ -z "$QUEUE_URL" ]; then
    aws sqs create-queue \
        --queue-name "$QUEUE_NAME" \
        --attributes \
        "VisibilityTimeout=960,MessageRetentionPeriod=345600,MaximumMessageSize=1048576,FifoQueue=true,ContentBasedDeduplication=true" \
        --region "$REGION"
    QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text)
    echo "  ✓ Created $QUEUE_NAME"
else
    echo "  Queue $QUEUE_NAME already exists"
fi

# Set queue policy for PaperScraper role
QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${QUEUE_NAME}"
QUEUE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPaperScraper",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${PAPER_SCRAPER_ROLE_ARN}"
      },
      "Action": [
        "sqs:ChangeMessageVisibility",
        "sqs:DeleteMessage",
        "sqs:ReceiveMessage"
      ],
      "Resource": "${QUEUE_ARN}"
    }
  ]
}
EOF
)

aws sqs set-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attributes "Policy=$(echo $QUEUE_POLICY | jq -c .)" \
    --region "$REGION" 2>/dev/null || true

# 2. judgeTestingQueue.fifo
echo "Creating judgeTestingQueue.fifo..."
QUEUE_NAME="judgeTestingQueue.fifo"
QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text 2>/dev/null || echo "")

if [ -z "$QUEUE_URL" ]; then
    aws sqs create-queue \
        --queue-name "$QUEUE_NAME" \
        --attributes \
        "VisibilityTimeout=960,MessageRetentionPeriod=345600,MaximumMessageSize=1048576,FifoQueue=true,ContentBasedDeduplication=true" \
        --region "$REGION"
    echo "  ✓ Created $QUEUE_NAME"
else
    echo "  Queue $QUEUE_NAME already exists"
fi

# Set queue policy
QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${QUEUE_NAME}"
QUEUE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowPaperScraper",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${PAPER_SCRAPER_ROLE_ARN}"
      },
      "Action": [
        "sqs:ChangeMessageVisibility",
        "sqs:DeleteMessage",
        "sqs:ReceiveMessage"
      ],
      "Resource": "${QUEUE_ARN}"
    }
  ]
}
EOF
)

QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text)
aws sqs set-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attributes "Policy=$(echo $QUEUE_POLICY | jq -c .)" \
    --region "$REGION" 2>/dev/null || true

# 3. trainium-execution.fifo
echo "Creating trainium-execution.fifo..."
QUEUE_NAME="trainium-execution.fifo"
QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text 2>/dev/null || echo "")

if [ -z "$QUEUE_URL" ]; then
    aws sqs create-queue \
        --queue-name "$QUEUE_NAME" \
        --attributes \
        "VisibilityTimeout=900,MessageRetentionPeriod=1209600,MaximumMessageSize=1048576,FifoQueue=true,ContentBasedDeduplication=true,DeduplicationScope=queue,FifoThroughputLimit=perQueue" \
        --region "$REGION"
    echo "  ✓ Created $QUEUE_NAME"
else
    echo "  Queue $QUEUE_NAME already exists"
fi

# 4. code-evaluation.fifo
echo "Creating code-evaluation.fifo..."
QUEUE_NAME="code-evaluation.fifo"
QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text 2>/dev/null || echo "")

if [ -z "$QUEUE_URL" ]; then
    aws sqs create-queue \
        --queue-name "$QUEUE_NAME" \
        --attributes \
        "VisibilityTimeout=60,MessageRetentionPeriod=1209600,MaximumMessageSize=1048576,FifoQueue=true,DeduplicationScope=messageGroup,FifoThroughputLimit=perMessageGroupId,ContentBasedDeduplication=false" \
        --region "$REGION"
    QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text)
    echo "  ✓ Created $QUEUE_NAME"
else
    echo "  Queue $QUEUE_NAME already exists"
fi

# Set queue policy for code-evaluation.fifo
QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${QUEUE_NAME}"
QUEUE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Id": "__default_policy_ID",
  "Statement": [
    {
      "Sid": "__owner_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::${ACCOUNT_ID}:root"
      },
      "Action": "SQS:*",
      "Resource": "${QUEUE_ARN}"
    },
    {
      "Sid": "__sender_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${PAPERS_JUDGE_ROLE_ARN}"
      },
      "Action": "SQS:SendMessage",
      "Resource": "${QUEUE_ARN}"
    },
    {
      "Sid": "__receiver_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${PAPERS_JUDGE_ROLE_ARN}"
      },
      "Action": [
        "SQS:ChangeMessageVisibility",
        "SQS:DeleteMessage",
        "SQS:ReceiveMessage"
      ],
      "Resource": "${QUEUE_ARN}"
    }
  ]
}
EOF
)

aws sqs set-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attributes "Policy=$(echo $QUEUE_POLICY | jq -c .)" \
    --region "$REGION" 2>/dev/null || true

# 5. code-evaluation-dlq.fifo
echo "Creating code-evaluation-dlq.fifo..."
QUEUE_NAME="code-evaluation-dlq.fifo"
QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text 2>/dev/null || echo "")

if [ -z "$QUEUE_URL" ]; then
    aws sqs create-queue \
        --queue-name "$QUEUE_NAME" \
        --attributes \
        "VisibilityTimeout=30,MessageRetentionPeriod=345600,MaximumMessageSize=1048576,FifoQueue=true,ContentBasedDeduplication=true,DeduplicationScope=queue,FifoThroughputLimit=perQueue" \
        --region "$REGION"
    echo "  ✓ Created $QUEUE_NAME"
else
    echo "  Queue $QUEUE_NAME already exists"
fi

# Set redrive policy for code-evaluation.fifo
DLQ_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:code-evaluation-dlq.fifo"
EVAL_QUEUE_URL=$(aws sqs get-queue-url --queue-name "code-evaluation.fifo" --query 'QueueUrl' --output text)
REDRIVE_POLICY=$(cat <<EOF
{
  "deadLetterTargetArn": "${DLQ_ARN}",
  "maxReceiveCount": 3
}
EOF
)

aws sqs set-queue-attributes \
    --queue-url "$EVAL_QUEUE_URL" \
    --attributes "RedrivePolicy=$(echo $REDRIVE_POLICY | jq -c .)" \
    --region "$REGION" 2>/dev/null || true

# 6. testingResults (Standard queue, not FIFO)
echo "Creating testingResults..."
QUEUE_NAME="testingResults"
QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text 2>/dev/null || echo "")

if [ -z "$QUEUE_URL" ]; then
    aws sqs create-queue \
        --queue-name "$QUEUE_NAME" \
        --attributes \
        "VisibilityTimeout=1800,MessageRetentionPeriod=345600,MaximumMessageSize=1048576" \
        --region "$REGION"
    QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --query 'QueueUrl' --output text)
    echo "  ✓ Created $QUEUE_NAME"
else
    echo "  Queue $QUEUE_NAME already exists"
fi

# Set queue policy for testingResults
QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${QUEUE_NAME}"
QUEUE_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Id": "__default_policy_ID",
  "Statement": [
    {
      "Sid": "__owner_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::${ACCOUNT_ID}:root"
      },
      "Action": "SQS:*",
      "Resource": "${QUEUE_ARN}"
    },
    {
      "Sid": "__sender_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${PAPERS_JUDGE_ROLE_ARN}"
      },
      "Action": "SQS:SendMessage",
      "Resource": "${QUEUE_ARN}"
    },
    {
      "Sid": "__receiver_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${PAPERS_JUDGE_ROLE_ARN}"
      },
      "Action": [
        "SQS:ChangeMessageVisibility",
        "SQS:DeleteMessage",
        "SQS:ReceiveMessage"
      ],
      "Resource": "${QUEUE_ARN}"
    }
  ]
}
EOF
)

aws sqs set-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attributes "Policy=$(echo $QUEUE_POLICY | jq -c .)" \
    --region "$REGION" 2>/dev/null || true

echo ""
echo "=== SQS Queues Setup Complete ==="

