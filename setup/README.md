# Annapurna Pipeline - AWS Infrastructure Setup

This directory contains scripts to set up all AWS infrastructure for the Annapurna research paper pipeline.

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **jq** installed (`brew install jq` on macOS, `apt-get install jq` on Linux)
3. **AWS Permissions** to create:
   - IAM roles and policies
   - Lambda functions
   - S3 buckets
   - SQS queues
   - Step Functions
   - DynamoDB tables
   - EventBridge rules

## Quick Start

Run the main setup script to set up everything:

```bash
cd setup
chmod +x *.sh
./setup_all.sh
```

Or run individual setup scripts in order:

```bash
./01_setup_iam_roles.sh
./02_setup_s3_buckets.sh
./03_setup_sqs_queues.sh
./04_setup_lambda_functions.sh
----> ./build_scraper.sh PaperScraperConferences
----> ./build_conference_wrapper.sh
----> ./build_scraper.sh PaperScraper_arxiv
----> ./build_judge.sh
----> ./build_judge_test.sh
----> ./build_cron_lambda.sh
./05_setup_step_functions.sh
./06_setup_dynamodb.sh
./07_setup_opensearch.sh  # Provides instructions only
```

## Setup Scripts Overview

### 01_setup_iam_roles.sh
Creates IAM roles:
- `PaperScraper` - For Lambda scraper functions (BasicExecution, S3 Full Access, SQS Full Access)
- `PapersJudge-role` - For PapersJudge Lambda (Bedrock Full Access, OpenSearch access)
- `annapurna-ecs-task-role` - For ECS containers (S3, SQS, OpenSearch, Bedrock access)
- `annapurna-ecs-execution-role` - For ECS task execution (ECR, CloudWatch Logs)

### 02_setup_s3_buckets.sh
Creates S3 buckets:
- `llm-research-papers` - Raw scraped papers
- `datasets-for-all-papers` - Dataset files (folders created, files need manual upload)
- `discarded-papers` - Rejected papers
- `papers-code-artifacts` - Generated code files
- `papers-test-outputs` - Test execution outputs
- `trainium-execution-results` - Trainium execution results

### 03_setup_sqs_queues.sh
Creates SQS queues:
- `researchQueue.fifo` - Papers pending evaluation (16 min timeout, 4 days retention)
- `judgeTestingQueue.fifo` - Test queue for PapersJudgeTest (16 min timeout, 4 days retention)
- `trainium-execution.fifo` - Trainium execution queue (15 min timeout, 14 days retention)
- `code-evaluation.fifo` - Code evaluation queue (1 min timeout, 14 days retention)
- `code-evaluation-dlq.fifo` - Dead letter queue for code-evaluation
- `testingResults` - Standard queue for test results (30 min timeout, 4 days retention)

### 04_setup_lambda_functions.sh
Configures Lambda functions (does NOT deploy code):
- `PaperScraperConferences` - Conference scraper (15 min timeout, 1024 MB memory)
- `conferenceWrapper` - Batch retrieval wrapper (5 min timeout, 1024 MB memory)
- `PaperScraper_arxiv` - ArXiv scraper with daily cron (15 min timeout, 1024 MB memory)
- `PapersJudge` - Paper evaluation (5 min timeout, 2000 MB memory, triggered by researchQueue.fifo)
- `PapersJudgeTest` - Test version of PapersJudge (5 min timeout, 2000 MB memory)
- `PapersCronJob` - Hourly cron job (5 min timeout, 512 MB memory)
- `PapersCodeGenerator-container` - ECS-based Lambda (configuration notes only)

**Note:** Lambda function code must be deployed separately using the deployment scripts in the `deployment/` directory.

### 05_setup_step_functions.sh
Creates Step Function:
- `conferenceScraper` - Orchestrates conference paper scraping

**Important:** Before creating the Step Function, you must update the Lambda ARNs in `conferenceScraper_definition.json`. The script will prompt you to confirm.

### 06_setup_dynamodb.sh
Creates DynamoDB table:
- `docRunErrors` - Stores execution errors (Pay-per-request billing)

### 07_setup_opensearch.sh
Provides instructions for manually setting up OpenSearch domain. OpenSearch is not created automatically due to cost and complexity considerations.

## Manual Steps Required

### 1. Update Step Function Definition
After running `05_setup_step_functions.sh`, edit `conferenceScraper_definition.json` and replace the Lambda ARNs with your actual ARNs:

```json
{
  "GetBatches": {
    "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:conferenceWrapper"
  },
  "ScrapeBatches": {
    "Iterator": {
      "States": {
        "ScrapeBatch": {
          "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:PaperScraperConferences"
        }
      }
    }
  }
}
```

Then update the Step Function:
```bash
aws stepfunctions update-state-machine \
  --state-machine-arn arn:aws:states:REGION:ACCOUNT:stateMachine:conferenceScraper \
  --definition file://conferenceScraper_definition.json
```

### 2. Create OpenSearch Domain
Follow the instructions in `07_setup_opensearch.sh` to:
1. Create OpenSearch domain named `research-papers`
2. Configure access policies
3. Create index `research-papers-v3` with KNN enabled
4. Update Lambda environment variables with the OpenSearch endpoint

### 3. Create and Deploy Lambda Function Code

**Important:** The deployment scripts (`build_*.sh`) only **update** existing Lambda functions. You must **create** the functions first, then deploy code to them.

#### Step 1: Build the Lambda packages
```bash
cd ../deployment
export AWS_PROFILE=your-profile-name  # Use your AWS profile

# Build all Lambda packages
./build_scraper.sh PaperScraperConferences
./build_conference_wrapper.sh
./build_scraper.sh PaperScraper_arxiv
./build_judge.sh
./build_judge_test.sh
./build_cron_lambda.sh
```

#### Step 2: Create the Lambda functions

Get your account ID and region:
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}
```

Create each Lambda function:

**PaperScraperConferences:**
```bash
cd deployment  # Make sure you're in the deployment directory
aws lambda create-function \
  --function-name PaperScraperConferences \
  --runtime python3.11 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PaperScraper \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://scraper_lambda.zip \
  --timeout 900 \
  --memory-size 1024 \
  --ephemeral-storage Size=1024 \
  --region ${REGION} \
  --environment "Variables={BUCKET_NAME=llm-research-papers-57185732,LOG_LEVEL=INFO,MAX_PAPERS=75,QUEUE_URL=https://sqs.${REGION}.amazonaws.com/${ACCOUNT_ID}/researchQueue.fifo,SOURCE=iclr,YEAR=2025}" \
  --description "Scrapes conference papers (ICLR, ICML, NeurIPS, MLSys)"
```

**conferenceWrapper:**
```bash
cd deployment  # Make sure you're in the deployment directory
aws lambda create-function \
  --function-name conferenceWrapper \
  --runtime python3.11 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PaperScraper \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://conference_wrapper.zip \
  --timeout 300 \
  --memory-size 1024 \
  --ephemeral-storage Size=1024 \
  --region ${REGION} \
  --description "Wrapper Lambda for conference batch retrieval"
```

**PaperScraper_arxiv:**
```bash
cd deployment  # Make sure you're in the deployment directory
aws lambda create-function \
  --function-name PaperScraper_arxiv \
  --runtime python3.11 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PaperScraper \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://scraper_lambda.zip \
  --timeout 900 \
  --memory-size 1024 \
  --ephemeral-storage Size=1024 \
  --region ${REGION} \
  --environment "Variables={BUCKET_NAME=llm-research-papers-57185732,LOG_LEVEL=INFO,MAX_PAPERS=6,QUEUE_URL=https://sqs.${REGION}.amazonaws.com/${ACCOUNT_ID}/researchQueue.fifo,SOURCE=arxiv,YEAR=2025}" \
  --description "Scrapes ArXiv papers daily"
```

**PapersJudge:**
```bash
cd deployment  # Make sure you're in the deployment directory
aws lambda create-function \
  --function-name PapersJudge \
  --runtime python3.11 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PapersJudge-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://judge.zip \
  --timeout 300 \
  --memory-size 2000 \
  --ephemeral-storage Size=512 \
  --region ${REGION} \
  --environment "Variables={BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0,OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com,OPENSEARCH_INDEX=research-papers-v3,SIMILARITY_THRESHOLD=0.80}" \
  --description "Evaluates paper relevance and novelty using Claude"
```

**PapersJudgeTest:**
```bash
cd deployment  # Make sure you're in the deployment directory
aws lambda create-function \
  --function-name PapersJudgeTest \
  --runtime python3.11 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PapersJudge-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://judge_test.zip \
  --timeout 300 \
  --memory-size 2000 \
  --ephemeral-storage Size=512 \
  --region ${REGION} \
  --environment "Variables={BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0,OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com,OPENSEARCH_INDEX=research-papers-v3,SIMILARITY_THRESHOLD=0.80,RESULTS_QUEUE_URL=https://sqs.${REGION}.amazonaws.com/${ACCOUNT_ID}/testingResults}" \
  --description "Test version of PapersJudge that sends results to queue"
```

**PapersCronJob:**
```bash
cd ..  # Go to project root (cron_lambda is in project root)
aws lambda create-function \
  --function-name PapersCronJob \
  --runtime python3.11 \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PapersCronJob-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://cron_lambda/build/cron_lambda.zip \
  --timeout 300 \
  --memory-size 512 \
  --ephemeral-storage Size=512 \
  --region ${REGION} \
  --environment "Variables={BATCH_SIZE_FOR_EXECUTION=10,CODE_BUCKET=papers-code-artifacts-57185732,CODE_EVAL_QUEUE_URL=https://sqs.${REGION}.amazonaws.com/${ACCOUNT_ID}/code-evaluation.fifo,CODE_GEN_LAMBDA_NAME=PapersCodeGenerator-container,MAX_CODE_GEN_CONCURRENT=5,MAX_PAPERS_PER_RUN=5,MAX_TRAINIUM_CONCURRENT=1,OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com,OPENSEARCH_INDEX=research-papers-v3,TRAINIUM_ENDPOINT=http://3.21.7.129:8000,TRAINIUM_EXECUTION_TIMEOUT=3600,TRAINIUM_INSTANCE_ID=i-0f0bf0de25aa4fd57a}" \
  --description "Cron job that manages paper processing pipeline"
```

**PapersCodeGenerator-container (Container-based Lambda):**

**Step 1: Create IAM Role**
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}

# Create the role
TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam create-role \
  --role-name PapersCodeGenerator-container-role \
  --assume-role-policy-document "$TRUST_POLICY" \
  --description "IAM role for PapersCodeGenerator-container Lambda function"

# Attach policies
aws iam attach-role-policy \
  --role-name PapersCodeGenerator-container-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name PapersCodeGenerator-container-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

aws iam attach-role-policy \
  --role-name PapersCodeGenerator-container-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
  --role-name PapersCodeGenerator-container-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess

# OpenSearch access (inline policy)
OPENSEARCH_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["es:ESHttpGet","es:ESHttpPost","es:ESHttpPut","es:DescribeElasticsearchDomain","es:DescribeElasticsearchDomains"],"Resource":"*"}]}'
aws iam put-role-policy \
  --role-name PapersCodeGenerator-container-role \
  --policy-name OpenSearchAccess \
  --policy-document "$OPENSEARCH_POLICY"
```

**Step 2: Ensure required file exists**
```bash
cd code_gen
# Copy the backup model file if needed
cp page_classifier_model.pkl.backup page_classifier_model.pkl
```

**Step 3: Build and push Docker image**
```bash
cd ../deployment
# This builds the image and pushes to ECR, but doesn't create the Lambda function
./build_lambda_container.sh
```

**Step 4: Create the Lambda function (if it doesn't exist)**
```bash
# Get the ECR image URI (use the digest from build script output, or use :latest tag)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}
# Option 1: Use digest (more reliable - get from build script output)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/paperscodegenerator-lambda@sha256:YOUR_DIGEST_HERE"
# Option 2: Use latest tag
# ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/paperscodegenerator-lambda:latest"

# Create the Lambda function
aws lambda create-function \
  --function-name PapersCodeGenerator-container \
  --package-type Image \
  --code ImageUri=${ECR_REPO} \
  --role arn:aws:iam::${ACCOUNT_ID}:role/PapersCodeGenerator-container-role \
  --timeout 900 \
  --memory-size 3008 \
  --ephemeral-storage Size=10240 \
  --region ${REGION} \
  --environment "Variables={BEDROCK_MODEL_ID=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0,CODE_BUCKET=papers-code-artifacts-57185732,FLASK_EXECUTE_ENDPOINT=http://3.21.7.129:8000/execute,OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com,OPENSEARCH_INDEX=research-papers-v3,SLACK_BOT_TOKEN=xoxb-552112250854-10119594925537-sqOfzVPjWTgcEswIWTRbKbax,SLACK_CHANNEL=ext-bdab-apl-research-papers,TRAINIUM_EXECUTION_QUEUE_URL=https://sqs.${REGION}.amazonaws.com/${ACCOUNT_ID}/trainium-execution.fifo}" \
  --description "Container-based Lambda for code generation"
```

**Note:** If the function already exists, the `build_lambda_container.sh` script will update it. If it doesn't exist, create it first using the command above, then you can use the build script for future updates.

**Note:** 
- Replace `${ACCOUNT_ID}` with your actual account ID (or use the variable)
- Replace `llm-research-papers-57185732` with your actual bucket name (check output from `02_setup_s3_buckets.sh`)
- Replace `papers-code-artifacts-57185732` with your actual code bucket name
- Replace `search-research-papers-XXXXX.us-east-1.es.amazonaws.com` with your actual OpenSearch endpoint
- Update `TRAINIUM_ENDPOINT` and `TRAINIUM_INSTANCE_ID` with your actual values

#### Step 3: Configure SQS Triggers

After creating the Lambda functions, set up SQS triggers to connect queues to functions:

**PapersJudge → researchQueue.fifo:**
```bash
aws lambda create-event-source-mapping \
  --function-name PapersJudge \
  --event-source-arn arn:aws:sqs:${REGION}:${ACCOUNT_ID}:researchQueue.fifo \
  --batch-size 1 \
  --scaling-config MaximumConcurrency=2 \
  --region ${REGION}
```

**PapersJudgeTest → judgeTestingQueue.fifo:**
```bash
aws lambda create-event-source-mapping \
  --function-name PapersJudgeTest \
  --event-source-arn arn:aws:sqs:${REGION}:${ACCOUNT_ID}:judgeTestingQueue.fifo \
  --batch-size 1 \
  --scaling-config MaximumConcurrency=2 \
  --region ${REGION}
```

**PapersCodeGenerator-container → code-evaluation.fifo:**
```bash
aws lambda create-event-source-mapping \
  --function-name PapersCodeGenerator-container \
  --event-source-arn arn:aws:sqs:${REGION}:${ACCOUNT_ID}:code-evaluation.fifo \
  --batch-size 10 \
  --region ${REGION}
```

**Note:** If `--scaling-config` is not supported in your AWS CLI version, you can set maximum concurrency later via the AWS Console or update the event source mapping:
```bash
# Get the UUID of the event source mapping first
UUID=$(aws lambda list-event-source-mappings \
  --function-name PapersJudge \
  --query 'EventSourceMappings[0].UUID' \
  --output text)

# Update with scaling config
aws lambda update-event-source-mapping \
  --uuid $UUID \
  --scaling-config MaximumConcurrency=2 \
  --region ${REGION}
```

#### Step 4: Configure Lambda functions (optional)
After creating the functions, you can re-run the setup script to configure triggers and cron jobs (it will skip creating triggers that already exist):
```bash
cd setup
AWS_PROFILE=your-profile-name ./04_setup_lambda_functions.sh
```

#### Step 4: Update function code (for future updates)
Once functions are created, you can use the build scripts to update code:
```bash
cd ../deployment
AWS_PROFILE=your-profile-name ./build_scraper.sh PaperScraperConferences
# etc...
```

### 4. Upload Dataset Files
Upload dataset files to `datasets-for-all-papers` S3 bucket. Expected structure:
```
datasets-for-all-papers/
├── cifar10/
├── cifar100/
├── fashion_mnist/
├── imdb/
├── mnist/
├── synthetic/
└── wikitext2/
```

### 5. Update Environment Variables
After deploying Lambda functions, update environment variables with actual values:
- `OPENSEARCH_ENDPOINT` - Your OpenSearch domain endpoint
- `QUEUE_URL` - Verify SQS queue URLs are correct (replace `ACCOUNT_ID` with your AWS account ID)
- `TRAINIUM_ENDPOINT` - Your Trainium instance endpoint
- `TRAINIUM_INSTANCE_ID` - Your EC2 instance ID

**Note:** All scripts automatically detect your AWS account ID. Queue URLs in environment variables should use your actual account ID, not the placeholder `ACCOUNT_ID` shown in examples.

## Environment Variables Reference

### PaperScraperConferences
```
BUCKET_NAME=llm-research-papers
LOG_LEVEL=INFO
MAX_PAPERS=75
QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/researchQueue.fifo
SOURCE=iclr
YEAR=2025
```

### PaperScraper_arxiv
```
BUCKET_NAME=llm-research-papers
LOG_LEVEL=INFO
MAX_PAPERS=6
QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/researchQueue.fifo
SOURCE=arxiv
YEAR=2025
```

### PapersJudge
```
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v3
SIMILARITY_THRESHOLD=0.80
```

### PapersJudgeTest
```
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v3
SIMILARITY_THRESHOLD=0.80
RESULTS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/testingResults
```

### PapersCronJob
```
BATCH_SIZE_FOR_EXECUTION=10
CODE_BUCKET=papers-code-artifacts
CODE_EVAL_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/code-evaluation.fifo
CODE_GEN_LAMBDA_NAME=PapersCodeGenerator-container
MAX_CODE_GEN_CONCURRENT=5
MAX_PAPERS_PER_RUN=5
MAX_TRAINIUM_CONCURRENT=1
OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v3
TRAINIUM_ENDPOINT=http://YOUR_INSTANCE_IP:8000
TRAINIUM_EXECUTION_TIMEOUT=3600
TRAINIUM_INSTANCE_ID=i-XXXXXXXXX
```

### PapersCodeGenerator-container (ECS)
```
BEDROCK_MODEL_ID=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0
CODE_BUCKET=papers-code-artifacts
FLASK_EXECUTE_ENDPOINT=http://YOUR_INSTANCE_IP:8000/execute
OPENSEARCH_ENDPOINT=search-research-papers-XXXXX.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v3
SLACK_BOT_TOKEN=your-slack-token
SLACK_CHANNEL=your-slack-channel
TRAINIUM_EXECUTION_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/trainium-execution.fifo
```

## Troubleshooting

### IAM Role Not Found
If you get "role not found" errors, make sure to run `01_setup_iam_roles.sh` first.

### Lambda Function Not Found
The deployment scripts (`build_*.sh`) only update existing Lambda functions. You must create the functions first using `aws lambda create-function` commands (see "Create and Deploy Lambda Function Code" section above). After creating the functions, you can use the build scripts to update code, and run `04_setup_lambda_functions.sh` to configure settings.

### SQS Queue Not Found
Make sure to run `03_setup_sqs_queues.sh` before `04_setup_lambda_functions.sh`.

### OpenSearch Endpoint
The OpenSearch endpoint in the scripts is a placeholder. Update it after creating your OpenSearch domain.

## Cleanup

To remove all resources (use with caution):

```bash
# Delete Lambda functions
aws lambda delete-function --function-name PaperScraperConferences
aws lambda delete-function --function-name conferenceWrapper
aws lambda delete-function --function-name PaperScraper_arxiv
aws lambda delete-function --function-name PapersJudge
aws lambda delete-function --function-name PapersJudgeTest
aws lambda delete-function --function-name PapersCronJob

# Delete Step Function
aws stepfunctions delete-state-machine --state-machine-arn arn:aws:states:REGION:ACCOUNT:stateMachine:conferenceScraper

# Delete SQS queues
aws sqs delete-queue --queue-url $(aws sqs get-queue-url --queue-name researchQueue.fifo --query QueueUrl --output text)
# Repeat for other queues...

# Delete S3 buckets (empty first!)
aws s3 rm s3://bucket-name --recursive
aws s3api delete-bucket --bucket bucket-name

# Delete DynamoDB table
aws dynamodb delete-table --table-name docRunErrors

# Delete IAM roles (detach policies first)
aws iam detach-role-policy --role-name PaperScraper --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
# Repeat for all policies...
aws iam delete-role --role-name PaperScraper
```

## Support

For issues or questions, refer to the main project README.md or contact the development team.

