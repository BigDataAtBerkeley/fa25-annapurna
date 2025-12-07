# Annapurna - Research Paper Pipeline

Automated pipeline for scraping, evaluating, generating code, and executing novel ML research papers on AWS Trainium.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Complete Pipeline Flow](#complete-pipeline-flow)
3. [Components](#components)
4. [Initial Setup](#initial-setup)
5. [Deployment](#deployment)
6. [Lambda Invocation](#lambda-invocation)
7. [Cron Job Management](#cron-job-management)
8. [Trainium Setup](#trainium-setup)
9. [Environment Variables](#environment-variables)
10. [Monitoring & Debugging](#monitoring--debugging)
11. [Local Development](#local-development)
12. [Cost Estimates](#cost-estimates)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH PAPER PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Conference      │  ICLR, ICML, NeurIPS, MLSys, ArXiv
│ Scrapers        │  (Lambda Functions)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ S3 Bucket       │  llm-research-papers
│ (Raw Papers)    │  All scraped PDFs stored permanently
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SQS Queue       │  researchQueue.fifo
│ (Paper Queue)   │  FIFO queue, one paper at a time
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PapersJudge     │  Lambda Function
│ Lambda          │  - Filters papers via Claude/Bedrock
│                 │  - Checks relevance & novelty (if both true, send to OpenSearch)
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│ OpenSearch       │  │ SQS Queue        │
│ research-papers  │  │ code-evaluation  │
│ -v2 or -v3      │  │ .fifo            │
│ (Metadata)       │  │ (Code Gen Queue) │
└────────┬────────┘  └────────┬─────────┘
         │                     │
         │                     │
         │                     ▼
         │            ┌─────────────────┐
         │            │ PapersCode       │
         │            │ Generator        │
         │            │ Lambda           │
         │            │ - Generates      │
         │            │   PyTorch code   │
         │            │ - Uses Claude    │
         │            │   via Bedrock    │
         │            └────────┬─────────┘
         │                     │
         │                     ├─────────────────┐
         │                     │                 │
         │                     ▼                 ▼
         │            ┌─────────────────┐  ┌─────────────────┐
         │            │ S3 Bucket        │  │ SQS Queue        │
         │            │ papers-code-     │  │ trainium-        │
         │            │ artifacts        │  │ execution.fifo   │
         │            │ (Generated Code) │  │ (Execution Queue)│
         │            └─────────────────┘  └────────┬─────────┘
         │                                          │
         │                                          ▼
         │                                 ┌─────────────────┐
         │                                 │ Trainium        │
         │                                 │ Executor        │
         │                                 │ (Flask App)     │
         │                                 │ - Executes code  │
         │                                 │ - Code review    │
         │                                 │ - Retries        │
         │                                 └────────┬────────┘
         │                                          │
         │                                          ▼
         │                                 ┌─────────────────┐
         │                                 │ Trainium        │
         │                                 │ Instance        │
         │                                 │ (trn1.2xlarge)  │
         │                                 │ - AWS Neuron    │
         │                                 │ - Hardware      │
         │                                 │   Acceleration  │
         │                                 └────────┬────────┘
         │                                          │
         │                                          ▼
         │                                 ┌─────────────────┐
         │                                 │ S3 Bucket       │
         │                                 │ papers-test-    │
         │                                 │ outputs         │
         │                                 │ (Results)       │
         │                                 └────────┬────────┘
         │                                          │
         └──────────────────────────────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ OpenSearch      │
                 │ (Updated with   │
                 │  results)       │
                 └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         AUTOMATION LAYER                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ PapersCronJob   │  Lambda Function (runs every 1 hour)
│ Lambda          │  - Queries OpenSearch for papers without execution
│                 │  - Sends papers to code-evaluation queue
│                 │  - Manages Trainium instance (start/stop)
│                 │  - Respects concurrency limits
└─────────────────┘
```

---

## Complete Pipeline Flow

### Step-by-Step Process

1. **Paper Scraping**
   - Conference scrapers (ICLR, ICML, NeurIPS, MLSys, ArXiv) run via Step Functions every day via cron job
   - Papers are downloaded and stored in S3 (`llm-research-papers` bucket)
   - Each paper is sent to `researchQueue.fifo` SQS queue

2. **Paper Filtering (Judge)**
   - `PapersJudge` Lambda is triggered by `researchQueue.fifo` (1 message at a time)
   - Uses Claude via Bedrock to evaluate paper relevance and novelty
   - Relevant & novel papers are indexed in OpenSearch (`research-papers-v3`)

3. **Code Generation**
   - `PapersCodeGenerator` Lambda is triggered by `code-evaluation.fifo` (batches of 10 or after 24 hours)
   - First step is running a trained logisitic classifier in the background to score pages by their relevance. Pages above a certain relevance threshold 
     are then sent as raw PDF bytes to Claude via Bedrock in chunks. 
   - Generates PyTorch code compatible with AWS Neuron SDK
   - `code_reviewer_0.py` is invoked, which makes sure initial code generated is actually compatible with Trn. 
   - Code is saved to S3 (`papers-code-artifacts` bucket) and sent to Slack in that paper's thread
   - Paper metadata is updated in OpenSearch
   - Code is sent to `trainium-execution.fifo` queue

4. **Code Execution (Trainium)**
   - Trainium executor (Flask app) processes messages from `trainium-execution.fifo`
   - The code is sent to an async Flask app (`app.py`) to avoid Lambda timeouts, which sits on the same EC2 instance as Trn. All execution occurs here. 
   - Code goes through a cycle of "code reviewers" where each reviewer tests the code on Trn, gets back errors, fixes them, and sends the updated code to the next code
     reviewer. Once no errors appear from the code (or after max 6 iterations), the final code is saved to `papers-code-artifacts`, sent to that 
     paper's thread in Slack, and tested on Trn. 
   - Code reviewers also pull past common execution errors from a DynamoDB database using `error_db.py`
   - After code finishes executing (succesfuly or otherwise), we update OpenSearch with execution results and metrics, as well as send results to that paper's Slack thread.

5. **Automation (Cron Job)**
   - `PapersCronJob` Lambda runs every 1 hour via EventBridge
   - Queries OpenSearch for papers without `executed_on_trn = true`
   - Sends papers to code generation queue (regenerates code for failed papers)
   - Manages Trainium instance lifecycle (starts when papers waiting, stops when idle)

---

## Components

### Lambda Functions

1. **PaperScraper_ICLR** - Scrapes ICLR conference papers
2. **PaperScraper_ICML** - Scrapes ICML conference papers
3. **PaperScraper_arxiv** - Scrapes ArXiv papers
4. **PaperScraper_NEURIPS** - Scrapes NeurIPS conference papers
5. **PaperScraper_MLSYS** - Scrapes MLSys conference papers
6. **conferenceWrapper** - Helper Lambda for Step Functions (retrieves batch sizes)
7. **PapersJudge** - Evaluates paper relevance using Claude/Bedrock
8. **PapersCodeGenerator** - Generates PyTorch code from papers (container-based recommended)
9. **PapersCodeTester** - Legacy batch dispatcher (deprecated - replaced by Trainium executor)
10. **PapersCronJob** - Automated pipeline management (runs every 1 hour)
11. **LogCleanupLambda** - Cleans up Lambda logs (optional)

### Compute Resources

- **Trainium Instance (trn1.2xlarge)** - AWS Neuron-powered instance for executing PyTorch code
  - Flask app runs on port 8000
  - Auto-starts/stops based on queue status
  - Hardware acceleration via AWS Neuron SDK

### S3 Buckets

- `llm-research-papers` - All scraped papers (PDFs) - permanent storage
- `papers-code-artifacts` - Generated PyTorch code files
- `papers-test-outputs` - Code execution logs, results, and metrics

### SQS Queues

- `researchQueue.fifo` - Papers pending evaluation (scraper → judge)
  - Batch size: 1 (process one at a time)
  - Visibility timeout: 900 seconds
  
- `code-evaluation.fifo` - Papers pending code generation (judge → code generator)
  - Batch size: 10 messages OR 24 hours (whichever comes first)
  - Visibility timeout: 900 seconds
  
- `trainium-execution.fifo` - Code pending execution (code generator → Trainium)
  - Processed by Trainium executor Flask app
  - FIFO queue for ordered processing

### OpenSearch Indexes

- `research-papers-v3` - Current index (used by cron job)
  - Stores paper metadata, code status, execution results, and metrics

### Step Functions

- `conferenceScraper` - State machine for parallel conference scraping
  - Uses MapState for concurrent batch processing
  - Invokes `conferenceWrapper` and `PaperScraper_ICLR` (unified scraper)

---

## Initial Setup

### Prerequisites

- AWS CLI configured with appropriate credentials
- Python 3.9+ installed
- Access to AWS services: Lambda, S3, SQS, OpenSearch, Bedrock, EC2, EventBridge
- SSH key for Trainium instance access

### First-Time Setup

```bash
# Make all scripts executable
chmod +x deployment/*.sh

# 1. Setup SQS queues and Lambda triggers
./deployment/setup_sqs_queues.sh

# 2. Setup pipeline infrastructure (S3 buckets, IAM policies)
./deployment/setup_pipeline.sh

# 3. Deploy all Lambda functions
./deployment/deploy_all.sh

# 4. Deploy conference wrapper (for Step Functions)
./deployment/build_conference_wrapper.sh

# 5. Deploy cron job Lambda
./deployment/build_cron_lambda.sh
./deployment/setup_cron_job.sh

# 6. Setup Trainium instance (see Trainium Setup section)
./deployment/deploy_trainium.sh /path/to/your-key.pem
```

---

## Deployment

### Deploy All Functions

```bash
./deployment/deploy_all.sh
```

This deploys:
- PapersJudge
- All scraper Lambdas (ICLR, ICML, ArXiv, NeurIPS, MLSys)

### Deploy Individual Components

#### Scrapers

```bash
# Deploy individual scraper
./deployment/build_scraper.sh PaperScraper_ICLR
./deployment/build_scraper.sh PaperScraper_ICML
./deployment/build_scraper.sh PaperScraper_arxiv
./deployment/build_scraper.sh PaperScraper_NEURIPS
./deployment/build_scraper.sh PaperScraper_MLSYS
```

#### Judge Lambda

```bash
./deployment/build_judge.sh
```

#### Code Generator Lambda

```bash
# Recommended: Container-based (fixes pymupdf issues)
./deployment/build_code_gen_lambda.sh

# Or: Docker-based deployment
./deployment/build_lambda_container.sh
```

#### Conference Wrapper

```bash
./deployment/build_conference_wrapper.sh
```

#### Cron Job Lambda

```bash
./deployment/build_cron_lambda.sh
./deployment/setup_cron_job.sh
```

#### Cleanup Lambda (Optional)

```bash
./deployment/build_cleanup.sh
```

---

## Lambda Invocation

### Trigger Scrapers

#### Using Step Functions (Recommended for Production)

```bash
# Execute scraping via Step Functions MapState
# This handles batching and concurrency automatically
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:478852001205:stateMachine:conferenceScraper \
  --name "test-60-papers-$(date +%s)" \
  --input '{
    "source": "iclr",
    "year": 2025,
    "search_term": "LLM",
    "batch_size": 30,
    "test_count": 300
  }'
```

**Parameters:**
- `source`: Conference source (`iclr`, `icml`, `neurips`, `mlsys`)
- `year`: Conference year (ex., `2025`)
- `search_term`: Search term for filtering papers (ex., `"LLM"`)
- `batch_size`: Number of papers per batch (ex., `30`)
- `test_count`: Total number of papers to scrape (optional, for testing)

#### Get Batch Sizes (Helper)

```bash
# Retrieve batch sizes for a conference
aws lambda invoke \
  --function-name conferenceWrapper \
  --payload '{
    "source": "iclr",
    "year": "2025",
    "batch_size": "30",
    "search_term": "LLM"
  }' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

cat scraper_output.json | python3 -m json.tool
```

#### Direct Lambda Invocation (Legacy)

```bash
# Scrape conference papers directly
# Note: PaperScraper_ICLR is the unified scraper for all conferences
# Replace "iclr" with "neurips", "mlsys", or "icml" as needed
aws lambda invoke \
  --function-name PaperScraper_ICLR \
  --payload '{
    "source": "iclr",
    "year": "2025",
    "batch_size": "5",
    "start_index": "100",
    "end_index": "105"
  }' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

```

### Generate Code for Papers

```bash
# Generate code by paper title
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{
    "action": "generate_by_title",
    "title": "ResNet",
    "max_papers": 1
  }' \
  --cli-binary-format raw-in-base64-out \
  response.json

# Generate code by paper ID
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{
    "action": "generate_by_id",
    "paper_id": "YOUR_PAPER_ID"
  }' \
  --cli-binary-format raw-in-base64-out \
  response.json

# Generate code for recent papers
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{
    "action": "generate_recent",
    "max_papers": 5
  }' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

### Invoke Cron Job Manually

```bash
# Invoke cron job for a random paper (sorts by ingest date, only runs for papers where executed_on_trn = False)
# Default: 1 paper (MAX_PAPERS_PER_RUN)
aws lambda invoke \
  --function-name PapersCronJob \
  --region us-east-1 \
  --payload '{}' \
  /tmp/cron_response.json && \
cat /tmp/cron_response.json | python3 -m json.tool

# Invoke cron job with 5 papers (for testing)
aws lambda invoke \
  --function-name PapersCronJob \
  --region us-east-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"max_papers": 5}' \
  /tmp/cron_response.json && \
cat /tmp/cron_response.json | python3 -m json.tool

# Invoke cron job for specific paper
aws lambda invoke \
  --function-name PapersCronJob \
  --region us-east-1 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"paper_id": "YOUR_PAPER_ID"}' \
  /tmp/cron_response.json && \
cat /tmp/cron_response.json | python3 -m json.tool
```

### Understanding SQS Queue Behavior and Lambda Timeouts

**How papers stay in queue until code generation finishes:**

1. **SQS Visibility Timeout**: When a message is received by Lambda, it becomes "invisible" for 900 seconds (15 minutes). This prevents other Lambdas from processing the same message.

2. **Lambda Processing**: The code generator Lambda processes papers **sequentially** (one at a time) to avoid Bedrock API throttling. Each paper typically takes 3-5 minutes to generate code and makes multiple Bedrock API calls (chunks, batches, final code generation).

3. **Automatic Retry on Timeout**: The Lambda monitors remaining time and handles timeouts gracefully:
   - **Before timeout**: Lambda checks remaining time before processing each paper
   - **60-second safety buffer**: Stops processing 60 seconds before timeout to ensure response can be returned
   - **SQS batch item failures**: Unprocessed papers are marked in `batchItemFailures` response
   - **Automatic retry**: SQS automatically retries only the unprocessed papers (not the successfully processed ones)
   - **No duplicates**: Successfully processed papers are NOT retried - only unprocessed papers are retried
   - **Trainium queue**: Successfully processed papers are automatically sent to Trainium execution queue before Lambda returns
   - **Example**: If Lambda processes 2 papers and times out:
     - Papers 1 & 2: Code generated → Sent to Trainium queue → SQS message deleted (not retried)
     - Papers 3, 4, 5: Marked for retry → SQS automatically retries these in next Lambda invocation

4. **Batch Processing**: The code generator Lambda is configured to:
   - Process up to 10 messages per batch (or wait 24 hours, whichever comes first)
   - Process papers **sequentially** (one at a time) to avoid Bedrock throttling
   - Lambda timeout: 900 seconds (15 minutes)
   - 2-second delay between papers for additional throttling mitigation

**Why Sequential Processing?**

Each paper makes multiple Bedrock API calls:
- PDF chunk summarization (multiple chunks per paper)
- Batch summarization (multiple batches)
- Final code generation

Processing 5 papers in parallel would result in 50-100+ concurrent Bedrock API calls, causing throttling errors. Sequential processing ensures we stay within Bedrock rate limits.

**To avoid Lambda timeouts with 5 papers:**

- **Option 1**: Increase Lambda timeout (recommended for 5 papers)
  ```bash
  aws lambda update-function-configuration \
    --function-name PapersCodeGenerator \
    --timeout 900  # 15 minutes (max) - allows ~3 papers
  ```
  **Note**: With 5 papers taking ~3-5 minutes each sequentially, you may need to process fewer papers per batch or accept that some will be retried.

- **Option 2**: Reduce SQS batch size (process fewer papers per Lambda invocation)
  ```bash
  # Find event source mapping UUID first
  aws lambda list-event-source-mappings \
    --function-name PapersCodeGenerator \
    --query 'EventSourceMappings[0].UUID' \
    --output text
  
  # Update to process 3 papers at a time instead of 10
  aws lambda update-event-source-mapping \
    --uuid <EVENT_SOURCE_MAPPING_UUID> \
    --batch-size 3
  ```
  This ensures each Lambda invocation processes fewer papers, reducing timeout risk.

- **Option 3**: Process papers in smaller batches via cron job
  - Set `MAX_PAPERS_PER_RUN=3` in cron Lambda environment variables
  - This processes 3 papers per cron run instead of 5

**Current Configuration:**
- SQS visibility timeout: 900 seconds (15 minutes)
- Lambda timeout: 900 seconds (15 minutes)
- Code generator processes: Papers **sequentially** (one at a time) to avoid Bedrock API throttling
- Expected time for 5 papers: ~15-25 minutes (each paper takes ~3-5 minutes)
- **Note**: Papers are processed sequentially because each paper makes multiple Bedrock API calls (chunks, batches, final code). Parallel processing would cause throttling errors.

**Note**: If Lambda times out, SQS automatically retries. The messages stay in the queue and become visible again, triggering a new Lambda invocation. This is handled automatically by AWS - no manual intervention needed.

---

## Cron Job Management

### Setup Cron Job

```bash
# Deploy cron Lambda
./deployment/build_cron_lambda.sh

# Setup EventBridge rule (runs every 1 hour)
./deployment/setup_cron_job.sh
```

### Control Cron Job

```bash
# Pause cron job
aws events disable-rule \
  --name papers-cron-job-1hour \
  --region us-east-1

# Resume cron job
aws events enable-rule \
  --name papers-cron-job-1hour \
  --region us-east-1

# Check cron job status
aws events describe-rule \
  --name papers-cron-job-1hour \
  --region us-east-1 \
  --query 'State' \
  --output text
```

### Cron Job Behavior

- Runs every 1 hour via EventBridge
- Queries OpenSearch for papers without `executed_on_trn = true`
- Processes up to `MAX_PAPERS_PER_RUN` papers per execution (default: 1)
- Sends papers to `code-evaluation.fifo` queue for code generation
- Manages Trainium instance lifecycle (auto-start/stop)
- Respects concurrency limits for code generation and Trainium execution

---

## Trainium Setup

### Deploy Trainium Executor

```bash
# Deploy Flask app to Trainium instance
./deployment/deploy_trainium.sh /path/to/your-key.pem

```

### Find Trainium Instance

```bash
# List Trainium instances
aws ec2 describe-instances \
  --region us-east-2 \
  --filters "Name=instance-type,Values=trn1.2xlarge" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
  --output table
```

### Manage Trainium Instance

```bash
# Start instance
aws ec2 start-instances \
  --region us-east-2 \
  --instance-ids i-0f0bf0de25aa4fd57

# Stop instance
aws ec2 stop-instances \
  --region us-east-2 \
  --instance-ids i-0f0bf0de25aa4fd57

# Check instance status
aws ec2 describe-instances \
  --region us-east-2 \
  --instance-ids i-0f0bf0de25aa4fd57 \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text
```

**Note:** The cron job automatically manages Trainium instance lifecycle based on queue status.

### Trainium Executor Endpoints

- `GET /health` - Health check endpoint
- `POST /execute` - Execute code asynchronously
- `GET /status` - Get execution status
- `POST /code_review` - Code review endpoint (internal)

---

## Environment Variables

### Scraper Lambdas (PaperScraper_*)

```bash
CONFERENCE=ICLR                    # Conference to scrape
CONFERENCE_YEAR=2025               # Year
MAX_PAPERS=3                       # Max papers to process
BUCKET_NAME=llm-research-papers    # S3 bucket for PDFs
QUEUE_URL=https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo
```

### PapersJudge Lambda

```bash
AWS_REGION=us-east-1
OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v2
CODE_EVAL_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

### PapersCodeGenerator Lambda

```bash
AWS_REGION=us-east-1
OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v2
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0  # Only 3-5 and 4-5 support direct PDF sends (Haiku won't work) 
CODE_BUCKET=papers-code-artifacts
TRAINIUM_EXECUTION_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo
USE_QUEUE_FOR_EXECUTION=true
ENABLE_EXECUTION_TESTING=false
TRAINIUM_EXECUTION_TIMEOUT=3600
```

### PapersCronJob Lambda

```bash
AWS_REGION=us-east-1
OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v3
CODE_EVAL_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo
TRAINIUM_ENDPOINT=http://YOUR_TRAINIUM_IP:8000
TRAINIUM_INSTANCE_ID=i-0f0bf0de25aa4fd57
TRAINIUM_EXECUTION_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo
MAX_CODE_GEN_CONCURRENT=5
MAX_TRAINIUM_CONCURRENT=1
MAX_PAPERS_PER_RUN=1
```

### Trainium Executor (Flask App)

```bash
# Set in .env file on Trainium instance
AWS_REGION=us-east-1
OPENSEARCH_ENDPOINT=https://your-opensearch-endpoint.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX=research-papers-v3
CODE_BUCKET=papers-code-artifacts
OUTPUTS_BUCKET=papers-test-outputs
SAGEMAKER_METRICS_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

---

## Monitoring & Debugging

### View Lambda Logs

```bash
# Scraper logs
aws logs tail /aws/lambda/PaperScraper_ICLR --since 5m --follow
aws logs tail /aws/lambda/PaperScraper_ICML --since 5m --follow
aws logs tail /aws/lambda/PaperScraper_arxiv --since 5m --follow
aws logs tail /aws/lambda/PaperScraper_NEURIPS --since 5m --follow
aws logs tail /aws/lambda/PaperScraper_MLSYS --since 5m --follow

# Judge logs
aws logs tail /aws/lambda/PapersJudge --since 15m --follow

# Code Generator logs
aws logs tail /aws/lambda/PapersCodeGenerator --since 5m --follow

# Cron Job logs
aws logs tail /aws/lambda/PapersCronJob --since 5m --follow
```

### Check SQS Queues

```bash
# Research queue (scraper → judge)
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible

# Code evaluation queue (judge → code generator)
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible

# Trainium execution queue (code generator → Trainium)
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible
```

### Check Pipeline Status

```bash
# Check all queue depths
echo "=== Queue Status ==="
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo \
  --attribute-names ApproximateNumberOfMessages \
  --query 'Attributes.ApproximateNumberOfMessages' --output text | xargs -I {} echo "researchQueue: {} messages"

CODE_EVAL=$(aws sqs get-queue-url --queue-name code-evaluation.fifo --query 'QueueUrl' --output text)
aws sqs get-queue-attributes --queue-url $CODE_EVAL \
  --attribute-names ApproximateNumberOfMessages \
  --query 'Attributes.ApproximateNumberOfMessages' --output text | xargs -I {} echo "code-evaluation: {} messages"

TRAINIUM_QUEUE=$(aws sqs get-queue-url --queue-name trainium-execution.fifo --query 'QueueUrl' --output text)
aws sqs get-queue-attributes --queue-url $TRAINIUM_QUEUE \
  --attribute-names ApproximateNumberOfMessages \
  --query 'Attributes.ApproximateNumberOfMessages' --output text | xargs -I {} echo "trainium-execution: {} messages"
```

### Check OpenSearch

```bash
# View all papers and their status
python debugging/check_opensearch.py

# Check OpenSearch field mapping
python debugging/check_opensearch_mapping.py

# Check specific paper
python debugging/check_error.py <paper_id>
```

### View Trainium Execution Logs

```bash
# SSH into Trainium instance
ssh -i /path/to/key.pem ec2-user@<TRAINIUM_IP>

# View Flask app logs
tail -f ~/trainium-executor/logs/trainium-executor.log

# View systemd service logs
sudo journalctl -u trainium-executor -f
```

### Purge SQS Queue (Emergency)

```bash

# Purge queue
aws sqs purge-queue \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo \
  --region us-east-1

# Verify queue is empty
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo \
  --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible \
  --region us-east-1
```

### CloudWatch Metrics

Training metrics from Trainium executions are automatically logged to CloudWatch:

```bash
# List all metrics for a paper
aws cloudwatch list-metrics \
  --namespace "Trainium/Training" \
  --dimensions Name=PaperId,Value=<PAPER_ID>

# Get statistics for a metric
aws cloudwatch get-metric-statistics \
  --namespace "Trainium/Training" \
  --metric-name training_loss \
  --dimensions Name=PaperId,Value=<PAPER_ID> \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum,Minimum
```

**CloudWatch Console:** Navigate to Metrics → `Trainium/Training` → Filter by PaperId

---

## Local Development

### Generate Code Locally

```bash
# Generate code by paper ID
python -m code_gen.main_handler generate_by_id \
  --paper-id "paper-id" \
  --save \
  --output-dir "code_gen/generated_code"

# Generate code by title
python -m code_gen.main_handler generate_by_title \
  --title "ResNet" \
  --save

# Generate code for recent papers
python -m code_gen.main_handler generate_recent --max-papers 5
```

### Download Code/Results from S3

```bash
# Download generated code from S3
python download_s3_code.py

# Download test results from S3
python download_test_results.py
```

---

## Cost Estimates

### Per 100 Papers

- **Scraping**: ~$0.10 (Lambda + S3)
- **Judging**: ~$0.05 (Lambda + Claude API)
- **Code Generation**: ~$3.00 (Bedrock/Claude API calls)
- **Code Testing**: ~$1.30 (Trainium trn1.2xlarge @ $1.34/hr for ~1hr + Lambda dispatch)
- **Storage**: ~$0.01 (S3 + OpenSearch)

**Total**: ~$4.46 per 100 papers

**Note**: Trainium costs assume batching reduces total execution time. On-demand pricing: trn1.2xlarge = $1.34/hour. The cron job automatically stops the instance when idle to minimize costs.

---

## Additional Resources

### Directory Structure

```
├── code_gen/                # Code generation module
│   ├── lambda_handler.py    # Lambda entry point
│   ├── chunked_generator.py # Chunked code generator
│   └── ...
├── trn_execute/             # Trainium execution module
│   ├── app.py              # Flask executor app
│   └── ...
├── judge_lambda/            # Judge Lambda function
├── scraper_lambda/          # Scraper Lambda functions
├── cron_lambda/             # Cron job Lambda
├── conference_wrapper/      # Conference wrapper Lambda
├── deployment/              # Deployment scripts
├── debugging/               # Debugging utilities
├── pipeline_for_delivery.py # Local pipeline script
└── storage_utils.py         # Storage utilities
```

### Key Features

- **Neuron SDK Integration**: Generated code uses `torch_xla` for Trainium compatibility
- **Hardware-Level Profiling**: Neuron Profiler captures hardware execution traces
- **CloudWatch Metrics**: Automatic logging of training and execution metrics
- **Slack Notifications**: Real-time updates on paper processing status
- **Automatic Retries**: Code review and automatic fixes for TRN compatibility
- **Cost Optimization**: Auto-start/stop Trainium instance based on queue status

---

## Troubleshooting

### Common Issues

1. **Lambda timeout errors**: Increase Lambda timeout (max 15 minutes for code generation)
2. **OpenSearch connection errors**: Check security groups and VPC configuration
3. **Trainium instance not starting**: Verify IAM permissions and instance tags
4. **Code generation failures**: Check Bedrock model ID and API permissions
5. **Queue messages stuck**: Check visibility timeout and Lambda concurrency limits

### Getting Help

- Check Lambda CloudWatch logs for detailed error messages
- Verify environment variables are set correctly
- Check SQS queue status and message visibility
- Review OpenSearch index mapping and field types
- Check Trainium executor logs for execution errors

---