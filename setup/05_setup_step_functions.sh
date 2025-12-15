#!/bin/bash
# Setup Step Functions for Annapurna Pipeline

set -e

echo "=== Setting up Step Functions ==="

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}

echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"

# Get Lambda function ARNs
get_lambda_arn() {
    local function_name=$1
    aws lambda get-function --function-name "$function_name" --region "$REGION" --query 'Configuration.FunctionArn' --output text 2>/dev/null || echo ""
}

CONFERENCE_WRAPPER_ARN=$(get_lambda_arn conferenceWrapper)
PAPER_SCRAPER_ARN=$(get_lambda_arn PaperScraperConferences)

if [ -z "$CONFERENCE_WRAPPER_ARN" ] || [ -z "$PAPER_SCRAPER_ARN" ]; then
    echo "Warning: Lambda functions not found. Please deploy Lambda functions first."
    echo "This script will create the Step Function definition, but you'll need to update ARNs manually."
fi

# Create IAM role for Step Functions
echo "Creating IAM role for Step Functions..."
STEP_FUNCTIONS_ROLE_NAME="conferenceScraper-StepFunctionsRole"

if aws iam get-role --role-name "$STEP_FUNCTIONS_ROLE_NAME" 2>/dev/null; then
    echo "  Role $STEP_FUNCTIONS_ROLE_NAME already exists"
else
    TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "states.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)
    
    aws iam create-role \
        --role-name "$STEP_FUNCTIONS_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "IAM role for conferenceScraper Step Function"
    
    # Attach policies
    aws iam attach-role-policy \
        --role-name "$STEP_FUNCTIONS_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess
    
    aws iam attach-role-policy \
        --role-name "$STEP_FUNCTIONS_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
    
    # X-Ray access
    aws iam attach-role-policy \
        --role-name "$STEP_FUNCTIONS_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess
    
    echo "  ✓ Created $STEP_FUNCTIONS_ROLE_NAME"
fi

STEP_FUNCTIONS_ROLE_ARN=$(aws iam get-role --role-name "$STEP_FUNCTIONS_ROLE_NAME" --query 'Role.Arn' --output text)

# Create Step Function definition
# Note: User needs to replace ARNs in the JSON file before uploading
echo ""
echo "Creating Step Function definition..."

# Use actual ARNs if available, otherwise use placeholder
if [ -z "$CONFERENCE_WRAPPER_ARN" ]; then
    CONFERENCE_WRAPPER_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:conferenceWrapper"
    echo "  Warning: conferenceWrapper Lambda not found. Using placeholder ARN."
    echo "  Update the ARN in the definition file after deploying the Lambda."
fi

if [ -z "$PAPER_SCRAPER_ARN" ]; then
    PAPER_SCRAPER_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:PaperScraperConferences"
    echo "  Warning: PaperScraperConferences Lambda not found. Using placeholder ARN."
    echo "  Update the ARN in the definition file after deploying the Lambda."
fi

STEP_FUNCTION_DEFINITION=$(cat <<EOF
{
  "Comment": "Conference Paper Scraper with Parallel Batches",
  "StartAt": "CountPapers",
  "States": {
    "CountPapers": {
      "Type": "Task",
      "Resource": "${CONFERENCE_WRAPPER_ARN}r",
      "ResultPath": "$.counter_result",
      "Next": "ProcessBatches",
      "Catch": [
        {
          "ErrorEquals": [
            "States.ALL"
          ],
          "ResultPath": "$.error",
          "Next": "CountFailed"
        }
      ]
    },
    "ProcessBatches": {
      "Type": "Map",
      "ItemsPath": "$.counter_result.batches",
      "MaxConcurrency": 100,
      "ResultPath": "$.batch_results",
      "Iterator": {
        "StartAt": "ScrapeBatch",
        "States": {
          "ScrapeBatch": {
            "Type": "Task",
            "Resource": "${PAPER_SCRAPER_ARN}",
            "TimeoutSeconds": 900,
            "Retry": [
              {
                "ErrorEquals": [
                  "States.TaskFailed",
                  "States.Timeout"
                ],
                "IntervalSeconds": 30,
                "MaxAttempts": 2,
                "BackoffRate": 2
              }
            ],
            "Catch": [
              {
                "ErrorEquals": [
                  "States.ALL"
                ],
                "ResultPath": "$.batch_error",
                "Next": "BatchFailed"
              }
            ],
            "End": true
          },
          "BatchFailed": {
            "Type": "Pass",
            "Result": {
              "status": "failed"
            },
            "End": true
          }
        }
      },
      "Next": "AggregateResults"
    },
    "AggregateResults": {
      "Type": "Pass",
      "Parameters": {
        "total_papers.$": "$.counter_result.total_papers",
        "num_batches.$": "$.counter_result.num_batches",
        "batch_results.$": "$.batch_results"
      },
      "End": true
    },
    "CountFailed": {
      "Type": "Fail",
      "Error": "CountPapersFailed",
      "Cause": "Failed to count papers from the source"
    }
  }
}
EOF
)

# Save definition to file
DEFINITION_FILE="conferenceScraper_definition.json"
echo "$STEP_FUNCTION_DEFINITION" > "$DEFINITION_FILE"
echo "  ✓ Created Step Function definition file: $DEFINITION_FILE"
echo ""
echo "  IMPORTANT: Please review and update the ARNs in $DEFINITION_FILE before creating the Step Function."
echo "  Replace the Lambda function ARNs with your actual ARNs."

# Create Step Function
STATE_MACHINE_NAME="conferenceScraper"

if aws stepfunctions describe-state-machine --state-machine-arn "arn:aws:states:${REGION}:${ACCOUNT_ID}:stateMachine:${STATE_MACHINE_NAME}" 2>/dev/null; then
    echo ""
    echo "  Step Function $STATE_MACHINE_NAME already exists."
    echo "  To update it, run:"
    echo "    aws stepfunctions update-state-machine \\"
    echo "      --state-machine-arn arn:aws:states:${REGION}:${ACCOUNT_ID}:stateMachine:${STATE_MACHINE_NAME} \\"
    echo "      --definition file://${DEFINITION_FILE} \\"
    echo "      --role-arn ${STEP_FUNCTIONS_ROLE_ARN}"
else
    echo ""
    echo "  Creating Step Function..."
    echo "  Note: Make sure to update ARNs in $DEFINITION_FILE first!"
    read -p "  Have you updated the ARNs in $DEFINITION_FILE? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        aws stepfunctions create-state-machine \
            --name "$STATE_MACHINE_NAME" \
            --definition "file://${DEFINITION_FILE}" \
            --role-arn "$STEP_FUNCTIONS_ROLE_ARN" \
            --region "$REGION" \
            --tracing-configuration enabled=true
        
        echo "  ✓ Created Step Function: $STATE_MACHINE_NAME"
    else
        echo "  Skipping Step Function creation. Update ARNs and run this script again."
    fi
fi

echo ""
echo "=== Step Functions Setup Complete ==="

