#!/bin/bash
#
# Setup SQS queues for the Annapurna pipeline
#

set -e

AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "========================================="
echo "SQS Queue Setup"
echo "========================================="
echo "Region: $AWS_REGION"
echo "Account: $ACCOUNT_ID"
echo ""

# Create researchQueue.fifo (scrapers → judge, ONE at a time)
echo "📝 Creating researchQueue.fifo..."
RESEARCH_QUEUE_URL=$(aws sqs create-queue \
  --queue-name researchQueue.fifo \
  --attributes '{
    "FifoQueue": "true",
    "ContentBasedDeduplication": "true",
    "MessageRetentionPeriod": "345600",
    "VisibilityTimeout": "900",
    "ReceiveMessageWaitTimeSeconds": "20"
  }' \
  --region $AWS_REGION \
  --query 'QueueUrl' \
  --output text 2>/dev/null || \
  aws sqs get-queue-url --queue-name researchQueue.fifo --region $AWS_REGION --query 'QueueUrl' --output text)

echo "✅ researchQueue.fifo: $RESEARCH_QUEUE_URL"

# Create code-evaluation.fifo (judge → code generator, batch of 10 or daily)
echo "📝 Creating code-evaluation.fifo..."
CODE_EVAL_QUEUE_URL=$(aws sqs create-queue \
  --queue-name code-evaluation.fifo \
  --attributes '{
    "FifoQueue": "true",
    "ContentBasedDeduplication": "true",
    "MessageRetentionPeriod": "345600",
    "VisibilityTimeout": "900",
    "ReceiveMessageWaitTimeSeconds": "20"
  }' \
  --region $AWS_REGION \
  --query 'QueueUrl' \
  --output text 2>/dev/null || \
  aws sqs get-queue-url --queue-name code-evaluation.fifo --region $AWS_REGION --query 'QueueUrl' --output text)

echo "✅ code-evaluation.fifo: $CODE_EVAL_QUEUE_URL"

# Create code-testing.fifo (code generator → tester, batch of 10)
echo "📝 Creating code-testing.fifo..."
CODE_TEST_QUEUE_URL=$(aws sqs create-queue \
  --queue-name code-testing.fifo \
  --attributes '{
    "FifoQueue": "true",
    "ContentBasedDeduplication": "true",
    "MessageRetentionPeriod": "345600",
    "VisibilityTimeout": "900",
    "ReceiveMessageWaitTimeSeconds": "20"
  }' \
  --region $AWS_REGION \
  --query 'QueueUrl' \
  --output text 2>/dev/null || \
  aws sqs get-queue-url --queue-name code-testing.fifo --region $AWS_REGION --query 'QueueUrl' --output text)

echo "✅ code-testing.fifo: $CODE_TEST_QUEUE_URL"

# Configure Lambda trigger for researchQueue.fifo → PapersJudge
echo ""
echo "📡 Configuring Lambda triggers..."

# === Configure Lambda Triggers ===

# 1. researchQueue.fifo → PapersJudge (NO BATCHING - process one at a time)
JUDGE_LAMBDA_ARN=$(aws lambda get-function \
  --function-name PapersJudge \
  --region $AWS_REGION \
  --query 'Configuration.FunctionArn' \
  --output text 2>/dev/null || echo "")

if [ -n "$JUDGE_LAMBDA_ARN" ]; then
    echo "🔗 Configuring trigger: researchQueue.fifo → PapersJudge"
    aws lambda create-event-source-mapping \
      --function-name PapersJudge \
      --batch-size 1 \
      --event-source-arn "arn:aws:sqs:$AWS_REGION:$ACCOUNT_ID:researchQueue.fifo" \
      --region $AWS_REGION 2>/dev/null || echo "  (Trigger already exists)"
    echo "✅ Configured (batch: 1 message, immediate processing)"
else
    echo "⚠️  PapersJudge Lambda not found"
fi

# 2. code-evaluation.fifo → PapersCodeGenerator (batch of 10)
CODEGEN_LAMBDA_ARN=$(aws lambda get-function \
  --function-name PapersCodeGenerator \
  --region $AWS_REGION \
  --query 'Configuration.FunctionArn' \
  --output text 2>/dev/null || echo "")

if [ -n "$CODEGEN_LAMBDA_ARN" ]; then
    echo "🔗 Configuring trigger: code-evaluation.fifo → PapersCodeGenerator"
    aws lambda create-event-source-mapping \
      --function-name PapersCodeGenerator \
      --batch-size 10 \
      --maximum-batching-window-in-seconds 86400 \
      --event-source-arn "arn:aws:sqs:$AWS_REGION:$ACCOUNT_ID:code-evaluation.fifo" \
      --region $AWS_REGION 2>/dev/null || echo "  (Trigger already exists)"
    echo "✅ Configured (batch: 10 messages OR 24 hours, whichever comes first)"
else
    echo "⚠️  PapersCodeGenerator Lambda not found"
fi

# 3. code-testing.fifo → PapersCodeTester (batch of 10)
TESTER_LAMBDA_ARN=$(aws lambda get-function \
  --function-name PapersCodeTester \
  --region $AWS_REGION \
  --query 'Configuration.FunctionArn' \
  --output text 2>/dev/null || echo "")

if [ -n "$TESTER_LAMBDA_ARN" ]; then
    echo "🔗 Configuring trigger: code-testing.fifo → PapersCodeTester"
    aws lambda create-event-source-mapping \
      --function-name PapersCodeTester \
      --batch-size 10 \
      --maximum-batching-window-in-seconds 3600 \
      --event-source-arn "arn:aws:sqs:$AWS_REGION:$ACCOUNT_ID:code-testing.fifo" \
      --region $AWS_REGION 2>/dev/null || echo "  (Trigger already exists)"
    echo "✅ Configured (batch: 10 messages OR 1 hour)"
else
    echo "⚠️  PapersCodeTester Lambda not found"
fi

echo ""
echo "========================================="
echo "✅ SQS Setup Complete!"
echo "========================================="
echo ""
echo "📊 CORRECTED Pipeline Flow:"
echo "  1. Scrapers → S3 + researchQueue.fifo (one paper at a time)"
echo "  2. PapersJudge (NO batching) → Bedrock → OpenSearch + code-evaluation.fifo"
echo "  3. code-evaluation.fifo (10 papers OR 24 hours) → PapersCodeGenerator"
echo "  4. PapersCodeGenerator → S3 + OpenSearch + code-testing.fifo"
echo "  5. code-testing.fifo (10 codes OR 1 hour) → PapersCodeTester → Trainium"
echo "  6. PapersCodeTester → S3 (papers-test-outputs) + OpenSearch"
echo ""
echo "📋 Queue URLs:"
echo "  researchQueue.fifo:     $RESEARCH_QUEUE_URL"
echo "  code-evaluation.fifo:   $CODE_EVAL_QUEUE_URL"
echo "  code-testing.fifo:      $CODE_TEST_QUEUE_URL"
echo ""
echo "🔧 Environment Variables:"
echo ""
echo "  PapersScraper_* Lambdas:"
echo "    QUEUE_URL=$RESEARCH_QUEUE_URL"
echo ""
echo "  PapersJudge Lambda:"
echo "    CODE_EVAL_QUEUE_URL=$CODE_EVAL_QUEUE_URL"
echo ""
echo "  PapersCodeGenerator Lambda:"
echo "    CODE_TEST_QUEUE_URL=$CODE_TEST_QUEUE_URL"
echo ""
echo "  PapersCodeTester Lambda:"
echo "    (triggered by code-testing.fifo)"
echo ""
echo "🔍 Monitor queues:"
echo "  aws sqs get-queue-attributes --queue-url $RESEARCH_QUEUE_URL --attribute-names ApproximateNumberOfMessages"
echo "  aws sqs get-queue-attributes --queue-url $CODE_EVAL_QUEUE_URL --attribute-names ApproximateNumberOfMessages"
echo "  aws sqs get-queue-attributes --queue-url $CODE_TEST_QUEUE_URL --attribute-names ApproximateNumberOfMessages"
echo ""

