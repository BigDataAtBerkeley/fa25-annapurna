#!/bin/bash
# Script to create IAM roles for ECS pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔐 Creating IAM Roles for ECS Pipeline${NC}"
echo ""

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${YELLOW}📋 AWS Account ID: ${ACCOUNT_ID}${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRUST_POLICY="${SCRIPT_DIR}/ecs-trust-policy.json"
TASK_POLICY="${SCRIPT_DIR}/ecs-task-policy.json"

# Check if policy files exist
if [ ! -f "$TRUST_POLICY" ]; then
    echo -e "${RED}❌ Error: ${TRUST_POLICY} not found${NC}"
    exit 1
fi

if [ ! -f "$TASK_POLICY" ]; then
    echo -e "${RED}❌ Error: ${TASK_POLICY} not found${NC}"
    exit 1
fi

# Step 1: Create Execution Role
echo -e "${YELLOW}Step 1: Creating ECS Task Execution Role...${NC}"
EXEC_ROLE_NAME="ecsTaskExecutionRole"

if aws iam get-role --role-name ${EXEC_ROLE_NAME} &>/dev/null; then
    echo -e "${YELLOW}   Role ${EXEC_ROLE_NAME} already exists, skipping creation${NC}"
else
    echo -e "${YELLOW}   Creating role: ${EXEC_ROLE_NAME}${NC}"
    aws iam create-role \
        --role-name ${EXEC_ROLE_NAME} \
        --assume-role-policy-document file://${TRUST_POLICY} \
        --description "ECS Task Execution Role for Annapurna Pipeline"
    echo -e "${GREEN}   ✓ Role created${NC}"
fi

# Attach AWS managed policy for ECS task execution
echo -e "${YELLOW}   Attaching AmazonECSTaskExecutionRolePolicy...${NC}"
aws iam attach-role-policy \
    --role-name ${EXEC_ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
echo -e "${GREEN}   ✓ Policy attached${NC}"

EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${EXEC_ROLE_NAME}"
echo -e "${GREEN}   Execution Role ARN: ${EXEC_ROLE_ARN}${NC}"
echo ""

# Step 2: Create Task Role
echo -e "${YELLOW}Step 2: Creating ECS Task Role...${NC}"
TASK_ROLE_NAME="ecsTaskRole"

if aws iam get-role --role-name ${TASK_ROLE_NAME} &>/dev/null; then
    echo -e "${YELLOW}   Role ${TASK_ROLE_NAME} already exists, skipping creation${NC}"
else
    echo -e "${YELLOW}   Creating role: ${TASK_ROLE_NAME}${NC}"
    aws iam create-role \
        --role-name ${TASK_ROLE_NAME} \
        --assume-role-policy-document file://${TRUST_POLICY} \
        --description "ECS Task Role for Annapurna Pipeline"
    echo -e "${GREEN}   ✓ Role created${NC}"
fi

# Attach custom policy
echo -e "${YELLOW}   Attaching AnnapurnaPipelinePolicy...${NC}"
aws iam put-role-policy \
    --role-name ${TASK_ROLE_NAME} \
    --policy-name AnnapurnaPipelinePolicy \
    --policy-document file://${TASK_POLICY}
echo -e "${GREEN}   ✓ Policy attached${NC}"

TASK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE_NAME}"
echo -e "${GREEN}   Task Role ARN: ${TASK_ROLE_ARN}${NC}"
echo ""

# Summary
echo -e "${GREEN}✅ IAM Roles Created Successfully!${NC}"
echo ""
echo -e "${GREEN}Summary:${NC}"
echo -e "  Execution Role: ${EXEC_ROLE_ARN}"
echo -e "  Task Role:      ${TASK_ROLE_ARN}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Update ecs-task-definition.json with these ARNs:"
echo -e "     - executionRoleArn: ${EXEC_ROLE_ARN}"
echo -e "     - taskRoleArn: ${TASK_ROLE_ARN}"
echo ""
echo -e "  2. Update ACCOUNT_ID in ecs-task-definition.json: ${ACCOUNT_ID}"

