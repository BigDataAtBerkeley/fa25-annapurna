#!/bin/bash
#
# Setup EventBridge rule to trigger Cron Lambda every 1 hour
#

set -e

AWS_REGION="${AWS_REGION:-us-east-1}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-PapersCronJob}"
RULE_NAME="${RULE_NAME:-papers-cron-job-1hour}"

echo "========================================="
echo "Setting up Cron Job for Papers Pipeline"
echo "========================================="
echo "Region: $AWS_REGION"
echo "Lambda: $LAMBDA_FUNCTION_NAME"
echo "Rule: $RULE_NAME"
echo ""

# Check if Lambda function exists
echo "🔍 Checking if Lambda function exists..."
LAMBDA_ARN=$(aws lambda get-function \
  --function-name "$LAMBDA_FUNCTION_NAME" \
  --region "$AWS_REGION" \
  --query 'Configuration.FunctionArn' \
  --output text 2>/dev/null || echo "")

if [ -z "$LAMBDA_ARN" ]; then
    echo "❌ Lambda function $LAMBDA_FUNCTION_NAME not found!"
    echo "   Please create the Lambda function first using build_cron_lambda.sh"
    exit 1
fi

echo "✅ Lambda function found: $LAMBDA_ARN"
echo ""

# Create or update EventBridge rule (runs every 1 hour)
echo "📅 Creating EventBridge rule (every 1 hour)..."
RULE_ARN=$(aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "rate(1 hour)" \
  --description "Trigger PapersCronJob Lambda every 1 hour to process papers" \
  --region "$AWS_REGION" \
  --query 'RuleArn' \
  --output text 2>/dev/null || \
  aws events describe-rule \
    --name "$RULE_NAME" \
    --region "$AWS_REGION" \
    --query 'Arn' \
    --output text)

echo "✅ EventBridge rule: $RULE_ARN"
echo ""

# Add Lambda permission for EventBridge
echo "🔐 Adding Lambda permission for EventBridge..."
aws lambda add-permission \
  --function-name "$LAMBDA_FUNCTION_NAME" \
  --statement-id "allow-eventbridge-${RULE_NAME}" \
  --action "lambda:InvokeFunction" \
  --principal "events.amazonaws.com" \
  --source-arn "$RULE_ARN" \
  --region "$AWS_REGION" 2>/dev/null || echo "  (Permission already exists)"
echo "✅ Permission added"
echo ""

# Add Lambda as target for the rule
echo "🎯 Adding Lambda as target for EventBridge rule..."
aws events put-targets \
  --rule "$RULE_NAME" \
  --targets "Id=1,Arn=$LAMBDA_ARN" \
  --region "$AWS_REGION" 2>/dev/null || echo "  (Target already exists)"
echo "✅ Target added"
echo ""

echo "========================================="
echo "✅ Cron Job Setup Complete!"
echo "========================================="
echo ""
echo "📋 Configuration:"
echo "  Rule: $RULE_NAME"
echo "  Schedule: Every 1 hour"
echo "  Lambda: $LAMBDA_FUNCTION_NAME"
echo ""
echo "🔍 Check rule status:"
echo "  aws events describe-rule --name $RULE_NAME --region $AWS_REGION"
echo ""
echo "📊 View recent invocations:"
echo "  aws logs tail /aws/lambda/$LAMBDA_FUNCTION_NAME --follow --region $AWS_REGION"
echo ""

