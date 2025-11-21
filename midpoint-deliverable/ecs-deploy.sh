#!/bin/bash
# ECS Deployment Script for Annapurna Pipeline

set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_NAME="annapurna-pipeline"
CLUSTER_NAME="annapurna-cluster"
SERVICE_NAME="annapurna-pipeline-service"
TASK_DEFINITION_FILE="ecs-task-definition.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting ECS Deployment${NC}"

# Step 1: Get AWS Account ID
echo -e "${YELLOW}📋 Getting AWS Account ID...${NC}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✓ Account ID: ${ACCOUNT_ID}${NC}"

# Step 2: Create ECR repository if it doesn't exist
echo -e "${YELLOW}📦 Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} 2>/dev/null; then
    echo -e "${YELLOW}Creating ECR repository: ${ECR_REPO_NAME}${NC}"
    aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}
    echo -e "${GREEN}✓ ECR repository created${NC}"
else
    echo -e "${GREEN}✓ ECR repository exists${NC}"
fi

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# Step 3: Login to ECR
echo -e "${YELLOW}🔐 Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}
echo -e "${GREEN}✓ Logged in to ECR${NC}"

# Step 4: Build Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker build -t ${ECR_REPO_NAME}:latest .
echo -e "${GREEN}✓ Docker image built${NC}"

# Step 5: Tag image
echo -e "${YELLOW}🏷️  Tagging image...${NC}"
docker tag ${ECR_REPO_NAME}:latest ${ECR_URI}:latest
docker tag ${ECR_REPO_NAME}:latest ${ECR_URI}:$(date +%Y%m%d-%H%M%S)
echo -e "${GREEN}✓ Image tagged${NC}"

# Step 6: Push image to ECR
echo -e "${YELLOW}📤 Pushing image to ECR...${NC}"
docker push ${ECR_URI}:latest
docker push ${ECR_URI}:$(date +%Y%m%d-%H%M%S)
echo -e "${GREEN}✓ Image pushed to ECR${NC}"

# Step 7: Update task definition with ECR URI
echo -e "${YELLOW}📝 Updating task definition...${NC}"
sed "s|YOUR_ECR_REPO_URI|${ECR_URI}|g; s|ACCOUNT_ID|${ACCOUNT_ID}|g" ${TASK_DEFINITION_FILE} > ${TASK_DEFINITION_FILE}.tmp
mv ${TASK_DEFINITION_FILE}.tmp ${TASK_DEFINITION_FILE}
echo -e "${GREEN}✓ Task definition updated${NC}"

# Step 8: Register task definition
echo -e "${YELLOW}📋 Registering task definition...${NC}"
TASK_DEF_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://${TASK_DEFINITION_FILE} \
    --region ${AWS_REGION} \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)
echo -e "${GREEN}✓ Task definition registered: ${TASK_DEF_ARN}${NC}"

# Step 9: Check if cluster exists
echo -e "${YELLOW}🔍 Checking ECS cluster...${NC}"
if ! aws ecs describe-clusters --clusters ${CLUSTER_NAME} --region ${AWS_REGION} --query 'clusters[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
    echo -e "${YELLOW}Creating ECS cluster: ${CLUSTER_NAME}${NC}"
    aws ecs create-cluster --cluster-name ${CLUSTER_NAME} --region ${AWS_REGION}
    echo -e "${GREEN}✓ Cluster created${NC}"
else
    echo -e "${GREEN}✓ Cluster exists${NC}"
fi

# Step 10: Check if service exists
echo -e "${YELLOW}🔍 Checking ECS service...${NC}"
if aws ecs describe-services --cluster ${CLUSTER_NAME} --services ${SERVICE_NAME} --region ${AWS_REGION} --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
    echo -e "${YELLOW}Updating ECS service...${NC}"
    aws ecs update-service \
        --cluster ${CLUSTER_NAME} \
        --service ${SERVICE_NAME} \
        --task-definition ${TASK_DEF_ARN} \
        --region ${AWS_REGION} \
        --force-new-deployment > /dev/null
    echo -e "${GREEN}✓ Service updated${NC}"
else
    echo -e "${YELLOW}⚠️  Service ${SERVICE_NAME} does not exist.${NC}"
    echo -e "${YELLOW}   Create it manually or use the CloudFormation template.${NC}"
fi

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}   Task Definition: ${TASK_DEF_ARN}${NC}"
echo -e "${GREEN}   Image: ${ECR_URI}:latest${NC}"

