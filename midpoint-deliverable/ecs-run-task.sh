#!/bin/bash
# Script to run ECS task with specific paper ID or query

set -e

# Configuration
CLUSTER_NAME="annapurna-cluster"
TASK_DEFINITION="annapurna-pipeline"
SUBNET_IDS="${SUBNET_IDS:-subnet-xxxxx,subnet-yyyyy}"  # Replace with your subnet IDs
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-xxxxx}"     # Replace with your security group ID

# Parse arguments
PAPER_ID=""
MAX_PAPERS=1
QUERY=""
RECENT_DAYS=30
USE_CHUNKED=false
ENABLE_EXECUTION_TESTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --paper-id)
            PAPER_ID="$2"
            shift 2
            ;;
        --max-papers)
            MAX_PAPERS="$2"
            shift 2
            ;;
        --query)
            QUERY="$2"
            shift 2
            ;;
        --recent-days)
            RECENT_DAYS="$2"
            shift 2
            ;;
        --use-chunked)
            USE_CHUNKED=true
            shift
            ;;
        --enable-execution-testing)
            ENABLE_EXECUTION_TESTING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command arguments
CMD_ARGS=()
if [ -n "$PAPER_ID" ]; then
    CMD_ARGS+=("--paper-id" "$PAPER_ID")
fi
CMD_ARGS+=("--max-papers" "$MAX_PAPERS")
if [ -n "$QUERY" ]; then
    CMD_ARGS+=("--query" "$QUERY")
else
    CMD_ARGS+=("--recent-days" "$RECENT_DAYS")
fi
if [ "$USE_CHUNKED" = true ]; then
    CMD_ARGS+=("--use-chunked")
fi
if [ "$ENABLE_EXECUTION_TESTING" = true ]; then
    CMD_ARGS+=("--enable-execution-testing")
fi

# Convert command args to JSON array
CMD_JSON=$(printf '%s\n' "${CMD_ARGS[@]}" | jq -R . | jq -s .)

# Run ECS task
echo "🚀 Running ECS task..."
echo "   Cluster: ${CLUSTER_NAME}"
echo "   Task Definition: ${TASK_DEFINITION}"
echo "   Command: ${CMD_ARGS[*]}"

TASK_ARN=$(aws ecs run-task \
    --cluster ${CLUSTER_NAME} \
    --task-definition ${TASK_DEFINITION} \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_IDS}],securityGroups=[${SECURITY_GROUP_ID}],assignPublicIp=ENABLED}" \
    --overrides "{\"containerOverrides\":[{\"name\":\"pipeline-container\",\"command\":${CMD_JSON}}]}" \
    --query 'tasks[0].taskArn' \
    --output text)

echo "✅ Task started: ${TASK_ARN}"
echo ""
echo "📊 Monitor task:"
echo "   aws ecs describe-tasks --cluster ${CLUSTER_NAME} --tasks ${TASK_ARN}"
echo ""
echo "📋 View logs:"
echo "   aws logs tail /ecs/annapurna-pipeline --follow"

