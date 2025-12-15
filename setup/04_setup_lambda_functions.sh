#!/bin/bash
# Setup Lambda Functions for Annapurna Pipeline

set -e

echo "=== Setting up Lambda Functions ==="

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}

echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"

# Get role ARNs
get_role_arn() {
    local role_name=$1
    aws iam get-role --role-name "$role_name" --query 'Role.Arn' --output text 2>/dev/null || echo ""
}

PAPER_SCRAPER_ROLE_ARN=$(get_role_arn PaperScraper)
PAPERS_JUDGE_ROLE_ARN=$(get_role_arn PapersJudge-role)

if [ -z "$PAPER_SCRAPER_ROLE_ARN" ] || [ -z "$PAPERS_JUDGE_ROLE_ARN" ]; then
    echo "Error: Required IAM roles not found. Please run 01_setup_iam_roles.sh first."
    exit 1
fi

# Get queue URLs
get_queue_url() {
    local queue_name=$1
    aws sqs get-queue-url --queue-name "$queue_name" --query 'QueueUrl' --output text 2>/dev/null || echo ""
}

RESEARCH_QUEUE_URL=$(get_queue_url researchQueue.fifo)
CODE_EVAL_QUEUE_URL=$(get_queue_url code-evaluation.fifo)
TESTING_QUEUE_URL=$(get_queue_url judgeTestingQueue.fifo)
TESTING_RESULTS_URL=$(get_queue_url testingResults)
TRAINIUM_QUEUE_URL=$(get_queue_url trainium-execution.fifo)

# Function to create or update Lambda
create_or_update_lambda() {
    local function_name=$1
    local role_arn=$2
    local handler=$3
    local timeout=$4
    local memory=$5
    local ephemeral_storage=$6
    local env_vars=$7
    local description=$8
    
    echo "Setting up Lambda: $function_name"
    
    # Check if function exists
    if aws lambda get-function --function-name "$function_name" --region "$REGION" 2>/dev/null; then
        echo "  Function $function_name already exists, updating configuration..."
        
        # Update configuration
        aws lambda update-function-configuration \
            --function-name "$function_name" \
            --role "$role_arn" \
            --timeout "$timeout" \
            --memory-size "$memory" \
            --ephemeral-storage "{\"Size\": $ephemeral_storage}" \
            --description "$description" \
            --region "$REGION" > /dev/null
        
        # Update environment variables if provided
        if [ -n "$env_vars" ]; then
            aws lambda update-function-configuration \
                --function-name "$function_name" \
                --environment "Variables={$env_vars}" \
                --region "$REGION" > /dev/null
        fi
        
        echo "  ✓ Updated $function_name"
    else
        echo "  Function $function_name does not exist."
        echo "  Note: You need to deploy the Lambda function code first using the deployment scripts."
        echo "  This script only sets up the configuration. Function will be created on first deployment."
    fi
}

# 1. PaperScraperConferences
echo ""
echo "=== PaperScraperConferences ==="
ENV_VARS="BUCKET_NAME=llm-research-papers,LOG_LEVEL=INFO,MAX_PAPERS=75,QUEUE_URL=${RESEARCH_QUEUE_URL},SOURCE=iclr,YEAR=2025"
create_or_update_lambda \
    "PaperScraperConferences" \
    "$PAPER_SCRAPER_ROLE_ARN" \
    "lambda_handler.lambda_handler" \
    900 \
    1024 \
    1024 \
    "$ENV_VARS" \
    "Scrapes conference papers (ICLR, ICML, NeurIPS, MLSys)"

# 2. conferenceWrapper
echo ""
echo "=== conferenceWrapper ==="
create_or_update_lambda \
    "conferenceWrapper" \
    "$PAPER_SCRAPER_ROLE_ARN" \
    "lambda_handler.lambda_handler" \
    300 \
    1024 \
    1024 \
    "" \
    "Wrapper Lambda for conference batch retrieval"

# 3. PaperScraper_arxiv
echo ""
echo "=== PaperScraper_arxiv ==="
ENV_VARS="BUCKET_NAME=llm-research-papers,LOG_LEVEL=INFO,MAX_PAPERS=6,QUEUE_URL=${RESEARCH_QUEUE_URL},SOURCE=arxiv,YEAR=2025"
create_or_update_lambda \
    "PaperScraper_arxiv" \
    "$PAPER_SCRAPER_ROLE_ARN" \
    "lambda_handler.lambda_handler" \
    900 \
    1024 \
    1024 \
    "$ENV_VARS" \
    "Scrapes ArXiv papers daily"

# Setup EventBridge rule for PaperScraper_arxiv (11pm UTC daily)
echo "  Setting up EventBridge cron for PaperScraper_arxiv..."
RULE_NAME="PaperScraperArxivDaily"

if aws events describe-rule --name "$RULE_NAME" --region "$REGION" 2>/dev/null; then
    echo "  Cron rule $RULE_NAME already exists"
else
    aws events put-rule \
        --name "$RULE_NAME" \
        --schedule-expression "cron(0 23 * * ? *)" \
        --description "Daily ArXiv scraper at 11pm UTC" \
        --region "$REGION" > /dev/null
    
    # Add Lambda permission
    aws lambda add-permission \
        --function-name "PaperScraper_arxiv" \
        --statement-id "EventBridge-${RULE_NAME}" \
        --action "lambda:InvokeFunction" \
        --principal events.amazonaws.com \
        --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
        --region "$REGION" 2>/dev/null || true
    
    # Add Lambda as target
    aws events put-targets \
        --rule "$RULE_NAME" \
        --targets "Id=1,Arn=arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:PaperScraper_arxiv" \
        --region "$REGION" > /dev/null
    
    echo "  ✓ Created cron rule for PaperScraper_arxiv"
fi

# 4. PapersJudge
echo ""
echo "=== PapersJudge ==="
ENV_VARS="BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0,OPENSEARCH_ENDPOINT=search-research-papers-uv3fxq76j5bkxq3bgp3nyfdtnm.us-east-1.es.amazonaws.com,OPENSEARCH_INDEX=research-papers-v3,SIMILARITY_THRESHOLD=0.80"
create_or_update_lambda \
    "PapersJudge" \
    "$PAPERS_JUDGE_ROLE_ARN" \
    "lambda_function.lambda_handler" \
    300 \
    2000 \
    512 \
    "$ENV_VARS" \
    "Evaluates paper relevance and novelty using Claude"

# Setup SQS trigger for PapersJudge
echo "  Setting up SQS trigger for PapersJudge..."
if [ -n "$RESEARCH_QUEUE_URL" ]; then
    # Remove existing event source mapping if any
    aws lambda list-event-source-mappings \
        --function-name "PapersJudge" \
        --region "$REGION" \
        --query 'EventSourceMappings[?EventSourceArn==`'"arn:aws:sqs:${REGION}:${ACCOUNT_ID}:researchQueue.fifo"'`].UUID' \
        --output text | while read uuid; do
        if [ -n "$uuid" ]; then
            aws lambda delete-event-source-mapping --uuid "$uuid" --region "$REGION" 2>/dev/null || true
        fi
    done
    
    # Create event source mapping
    QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:researchQueue.fifo"
    aws lambda create-event-source-mapping \
        --function-name "PapersJudge" \
        --event-source-arn "$QUEUE_ARN" \
        --batch-size 1 \
        --maximum-concurrency 2 \
        --region "$REGION" 2>/dev/null || echo "  Event source mapping may already exist or function not deployed yet"
else
    echo "  Warning: researchQueue.fifo not found. Run 03_setup_sqs_queues.sh first."
fi

# 5. PapersJudgeTest
echo ""
echo "=== PapersJudgeTest ==="
ENV_VARS="BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0,OPENSEARCH_ENDPOINT=search-research-papers-uv3fxq76j5bkxq3bgp3nyfdtnm.us-east-1.es.amazonaws.com,OPENSEARCH_INDEX=research-papers-v3,SIMILARITY_THRESHOLD=0.80,RESULTS_QUEUE_URL=${TESTING_RESULTS_URL}"
create_or_update_lambda \
    "PapersJudgeTest" \
    "$PAPERS_JUDGE_ROLE_ARN" \
    "lambda_function.lambda_handler" \
    300 \
    2000 \
    512 \
    "$ENV_VARS" \
    "Test version of PapersJudge that sends results to queue"

# Setup SQS trigger for PapersJudgeTest
echo "  Setting up SQS trigger for PapersJudgeTest..."
if [ -n "$TESTING_QUEUE_URL" ]; then
    QUEUE_ARN="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:judgeTestingQueue.fifo"
    aws lambda create-event-source-mapping \
        --function-name "PapersJudgeTest" \
        --event-source-arn "$QUEUE_ARN" \
        --batch-size 1 \
        --maximum-concurrency 2 \
        --region "$REGION" 2>/dev/null || echo "  Event source mapping may already exist or function not deployed yet"
else
    echo "  Warning: judgeTestingQueue.fifo not found. Run 03_setup_sqs_queues.sh first."
fi

# 6. PapersCronJob
echo ""
echo "=== PapersCronJob ==="
ENV_VARS="BATCH_SIZE_FOR_EXECUTION=10,CODE_BUCKET=papers-code-artifacts,CODE_EVAL_QUEUE_URL=${CODE_EVAL_QUEUE_URL},CODE_GEN_LAMBDA_NAME=PapersCodeGenerator-container,MAX_CODE_GEN_CONCURRENT=5,MAX_PAPERS_PER_RUN=5,MAX_TRAINIUM_CONCURRENT=1,OPENSEARCH_ENDPOINT=search-research-papers-uv3fxq76j5bkxq3bgp3nyfdtnm.us-east-1.es.amazonaws.com,OPENSEARCH_INDEX=research-papers-v3,TRAINIUM_ENDPOINT=http://3.21.7.129:8000,TRAINIUM_EXECUTION_TIMEOUT=3600,TRAINIUM_INSTANCE_ID=i-0f0bf0de25aa4fd57a"

# Get PapersCronJob role ARN (may need to be created separately)
# Create PapersCronJob role if it doesn't exist
PAPERS_CRON_ROLE_ARN=$(get_role_arn PapersCronJob-role)
if [ -z "$PAPERS_CRON_ROLE_ARN" ]; then
    echo "  Creating PapersCronJob-role..."
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
    aws iam create-role \
        --role-name PapersCronJob-role \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "IAM role for PapersCronJob Lambda function" > /dev/null
    
    aws iam attach-role-policy \
        --role-name PapersCronJob-role \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    
    aws iam attach-role-policy \
        --role-name PapersCronJob-role \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    
    aws iam attach-role-policy \
        --role-name PapersCronJob-role \
        --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess
    
    # OpenSearch access
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
        --role-name PapersCronJob-role \
        --policy-name OpenSearchAccess \
        --policy-document "$OPENSEARCH_POLICY"
    
    # EC2 access for instance management
    aws iam attach-role-policy \
        --role-name PapersCronJob-role \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess
    
    # Lambda invoke for code generation
    aws iam attach-role-policy \
        --role-name PapersCronJob-role \
        --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess
    
    # CloudWatch for metrics
    aws iam attach-role-policy \
        --role-name PapersCronJob-role \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess
    
    PAPERS_CRON_ROLE_ARN=$(get_role_arn PapersCronJob-role)
    echo "  ✓ Created PapersCronJob-role"
fi

create_or_update_lambda \
    "PapersCronJob" \
    "$PAPERS_CRON_ROLE_ARN" \
    "lambda_function.lambda_handler" \
    300 \
    512 \
    512 \
    "$ENV_VARS" \
    "Cron job that manages paper processing pipeline"

# Setup EventBridge rule for PapersCronJob (every hour)
echo "  Setting up EventBridge cron for PapersCronJob..."
RULE_NAME="PapersCronJobHourly"

if aws events describe-rule --name "$RULE_NAME" --region "$REGION" 2>/dev/null; then
    echo "  Cron rule $RULE_NAME already exists"
else
    aws events put-rule \
        --name "$RULE_NAME" \
        --schedule-expression "rate(1 hour)" \
        --description "Hourly cron job for paper processing" \
        --region "$REGION" > /dev/null
    
    # Add Lambda permission
    aws lambda add-permission \
        --function-name "PapersCronJob" \
        --statement-id "EventBridge-${RULE_NAME}" \
        --action "lambda:InvokeFunction" \
        --principal events.amazonaws.com \
        --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}" \
        --region "$REGION" 2>/dev/null || true
    
    # Add Lambda as target
    aws events put-targets \
        --rule "$RULE_NAME" \
        --targets "Id=1,Arn=arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:PapersCronJob" \
        --region "$REGION" > /dev/null
    
    echo "  ✓ Created cron rule for PapersCronJob"
fi

# 7. PapersCodeGenerator-container (ECS-based, handled separately)
echo ""
echo "=== PapersCodeGenerator-container ==="
echo "  Note: PapersCodeGenerator-container is an ECS-based Lambda function."
echo "  Configuration will be handled by ECS task definition."
echo "  Environment variables:"
echo "    BEDROCK_MODEL_ID=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
echo "    CODE_BUCKET=papers-code-artifacts"
echo "    FLASK_EXECUTE_ENDPOINT=http://3.21.7.129:8000/execute"
echo "    OPENSEARCH_ENDPOINT=search-research-papers-uv3fxq76j5bkxq3bgp3nyfdtnm.us-east-1.es.amazonaws.com"
echo "    OPENSEARCH_INDEX=research-papers-v3"
echo "    SLACK_BOT_TOKEN=xoxb-552112250854-10119594925537-sqOfzVPjWTgcEswIWTRbKbax"
echo "    SLACK_CHANNEL=ext-bdab-apl-research-papers"
echo "    TRAINIUM_EXECUTION_QUEUE_URL=${TRAINIUM_QUEUE_URL}"

echo ""
echo "=== Lambda Functions Setup Complete ==="
echo ""
echo "Note: Lambda function code must be deployed separately using the deployment scripts."
echo "This script only configures the Lambda function settings and triggers."

