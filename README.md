# Annapurna - Research Paper Pipeline

Automated pipeline for scraping, evaluating, and generating code from ML research papers.

---

## Architecture Overview

```
Papers Scrapers (separate lambda functions for ICLR, ICML, arXiv,NEURIPS)
    ↓
S3 (llm-research-papers) --> all raw papers stored here (literally every single paper scraped. no papers ever deleted from this bucket)
    ↓
SQS (researchQueue.fifo) --> every paper from S3 thats initially scraped gets sent here. should be empty 99% of the time bc papersJudge Lambda is invoked every time a paper hits the front of this queue
    ↓
PapersJudge Lambda (filters relevant papers) --> every paper in researchQueue.fifo gets sent here, where a call to Claude via Bedrock determines if it should be indexed in OpenSearch (based on relevance and novelty)
    ↓
OpenSearch (research-papers-v2) --> intially, filtered papers and their metadata stored here (based on what Claude determined relevant and novel)
    ↓
PapersCodeGenerator Lambda (generates PyTorch code) --> every paper from OpenSearch gets sent here where a call to Clauda via Bedrock generates its code & grabs any dataset mentioned
    ↓
S3 (papers-code-artifacts) + OpenSearch (code metadata) --> code files from generated code gets stored here in S3 (PyTorch code)
    ↓
SQS (code-evaluation.fifo) --> every code file from papers-code-artifacts S3 gets sent here (accumulates in batches of 10)
    ↓
PapersCodeTester Lambda (batch dispatcher) --> triggered when 10 papers accumulate in SQS, downloads code from S3, batches them together, and sends to Trainium instance
    ↓
Trainium Instance (trn1.2xlarge) --> executes PyTorch code with AWS Neuron SDK and hardware acceleration, returns results to Lambda
    ↓
S3 (papers-test-outputs) + OpenSearch (test results) --> execution results from Trainium get stored in S3, then attached to the original paper in OpenSearch
```

---

## Components

### **Lambda Functions**
1. **PaperScraper_ICLR** - Scrapes ICLR papers
2. **PaperScraper_ICML** - Scrapes ICML papers
3. **PaperScraper_arxiv** - Scrapes ArXiv papers
4. **PaperScraper_NEURIPS** - Scrapes NeurIPS papers
5. **PaperScraper_MLSYS** - Scrapes MLSys papers
6. **PapersJudge** - Evaluates paper relevance
7. **PapersCodeGenerator** - Generates PyTorch code from papers
8. **PapersCodeTester** - Batches code (10 at a time) and dispatches to Trainium instance for execution
9. **LogCleanupLambda** - Cleans up Lambda logs (optional)

### **Compute Resources**
- **Trainium Instance (trn1.2xlarge)** - AWS Neuron-powered instance for executing PyTorch code with hardware acceleration

### **S3 Buckets**
- `llm-research-papers` - Scraped papers (PDFs)
- `papers-code-artifacts` - Generated PyTorch code
- `papers-test-outputs` - Code execution logs & results

### **SQS Queues**
- `researchQueue.fifo` - Papers pending evaluation
- `code-evaluation.fifo` - Code pending testing
- `code-testing.fifo` - testing on trn instances (10 )

### **OpenSearch Index**
- `research-papers-v2` - All papers with metadata, code status, and test results

---

## Quick Start

### Initial Setup (First Time)
```bash
# Make all scripts executable
chmod +x deployment/*.sh

# 1. Setup infrastructure (SQS queues, S3 buckets, IAM policies)
./deployment/setup_sqs_queues.sh
./deployment/setup_pipeline.sh

# 2. Deploy all Lambda functions
./deployment/deploy_all.sh
```

### Deploy All Functions (After Initial Setup)
```bash
./deployment/deploy_all.sh
```

### Deploy Individual Components
```bash
# Scrapers
./deployment/build_scraper.sh PaperScraper_ICLR
./deployment/build_scraper.sh PaperScraper_ICML
./deployment/build_scraper.sh PaperScraper_arxiv
./deployment/build_scraper.sh PaperScraper_NEURIPS
./deployment/build_scraper.sh PaperScraper_MLSYS

# Judge
./deployment/build_judge.sh

# Code Generator
./deployment/build_code_gen_lambda.sh

# Code Tester
./deployment/build_test_lambda.sh

# Cleanup Lambda (optional)
./deployment/build_cleanup.sh
```

---

## Testing & Usage

### Trigger Scrapers Manually

```bash

# Execute Scraping of Conference Papers via MapState. This should be used in production.
# If it's failing above 10 batches, check concurenncy limit. It's currently set to 10 but can be maxed at 1000.
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:478852001205:stateMachine:conferenceScraper \
  --name "test-60-papers-$(date +%s)" \
  --input '{"source": "iclr", "year": 2025, "search_term": "LLM", "batch_size": 30, "test_count": 300}'

# Retrive Batch Sizes for conferences
aws lambda invoke \
  --function-name conferenceWrapper \
  --payload '{"source": "iclr", "year": "2025", "batch_size": "30", "search_term": "LLM"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

# Scrape conference papers - replace "iclr" with "neurips", "mlsys", or "icml". 
# DO NOT CHANGE "PaperScraper_ICLR" - this is scraper lambda for all conferences, the naming convention just hasn't been updated.
aws lambda invoke \
  --function-name PaperScraper_ICLR \
  --payload '{"source": "iclr", "year": "2025", "batch_size": "5", "start_index": "100", "end_index": "105"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

# OLD INVOCATION FUNCTIONS - these are still deployed, but we highley reccomend using the command above.
aws lambda invoke \
  --function-name PaperScraper_ICML \
  --payload '{"MAX_PAPERS": "5"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

aws lambda invoke \
  --function-name PaperScraper_arxiv \
  --payload '{"MAX_PAPERS": "5"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

aws lambda invoke \
  --function-name PaperScraper_NEURIPS \
  --payload '{"MAX_PAPERS": "5"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

aws lambda invoke \
  --function-name PaperScraper_MLSYS \
  --payload '{"MAX_PAPERS": "5"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json
```

### Generate Code for Papers

```bash
# Generate code by paper title
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{"action":"generate_by_title","title":"{{INSERT TITLE}}","max_papers":1}' \
  --cli-binary-format raw-in-base64-out \
  response.json

# Generate code by paper ID
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{"action":"generate_by_id","paper_id":"{{INSERT PAPERID}}"}' \
  --cli-binary-format raw-in-base64-out \
  response.json

# Generate code for recent papers (recency defined by days)
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{"action":"generate_recent","max_papers":5}' \
  --cli-binary-format raw-in-base64-out \
  response.json

```

### Test Generated Code on Trainium

Test a specific paper's generated code directly on Trainium (bypasses full pipeline):

```bash
# Test by paper ID (downloads code from S3)
python test_code_on_trainium.py --paper-id <PAPER_ID>

# Test local code file
python test_code_on_trainium.py --file generated_code/my_code.py --paper-id <PAPER_ID>

# Test with custom timeout (default is 600s)
python test_code_on_trainium.py --paper-id <PAPER_ID> --timeout 900

# Test without saving results (for debugging)
python test_code_on_trainium.py --paper-id <PAPER_ID> --no-save

# Example with actual paper ID:
python test_code_on_trainium.py --paper-id 6-j63JkBP8oloYi_8CJH
```

**What it does:**
- Downloads code from S3 (or uses local file)
- Sends to Trainium for execution
- Saves stdout/stderr/metrics to S3
- Updates OpenSearch with test results
- Displays execution results and metrics

**Requirements:**
- Trainium instance running (script will auto-start if stopped)
- Flask app deployed on Trainium (`./deployment/deploy_trainium.sh`)
- `TRAINIUM_ENDPOINT` in `.env` (and optionally `TRAINIUM_INSTANCE_ID` for auto-start)

### Test Code and View SageMaker Metrics

Test code on Trainium and automatically view metrics logged to CloudWatch:

```bash
# Test code and view metrics (by paper ID from S3)
python test_and_view_metrics.py --paper-id 6-j63JkBP8oloYi_8CJH

# Test local code file and view metrics
python test_and_view_metrics.py --file generated_code/TurboAttention_MODIFIED_with_dataset_loader.py --paper-id 6-j63JkBP8oloYi_8CJH

# View existing metrics without re-running (skip execution)
python test_and_view_metrics.py --paper-id 6-j63JkBP8oloYi_8CJH --skip-execution
```

**What it does:**
- Executes code on Trainium (same as `test_code_on_trainium.py`)
- Automatically logs training metrics to CloudWatch (namespace: `Trainium/Training`)
- Waits for metrics to appear and displays summary
- Shows CloudWatch console link and CLI commands

**View metrics in CloudWatch:**
```bash
# List all metrics for a paper
aws cloudwatch list-metrics --namespace "Trainium/Training" --dimensions Name=PaperId,Value=<PAPER_ID>

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

### Grabbing code locally from S3

```bash
1. python download_s3_code.py
--> this will ask you which code files to download from S3 (choose which you want)
2. Check generated_code/
---> this should contain (a) pyTorch code that was generated and (b) metadata about the paper & its code 

```

### CLI Usage (Local)

```bash
# Generate code by paper ID (local)
python -m code_gen.main_handler generate_by_id \
  --paper-id "paper-id" \
  --save \
  --output-dir "code_gen/generated_code"

# Generate code by title (local)
python -m code_gen.main_handler generate_by_title \
  --title "ResNet" \
  --save

# Generate code for 5 most recent papers
python -m code_gen.main_handler generate_recent --max-papers 5
```

---

## Monitoring (looking at logs after running scrapers/code_gen/etc locally)

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
a
# Code Generator logs
aws logs tail /aws/lambda/PapersCodeGenerator --since 5m --follow

# Code Tester logs
aws logs tail /aws/lambda/PapersCodeTester --since 5m --follow
```

### Check SQS Queues

```bash
# Papers queue (scraper → judge)
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo \
  --attribute-names ApproximateNumberOfMessages

# Code testing queue (generator → tester)
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/code-evaluation.fifo \
  --attribute-names ApproximateNumberOfMessages
```

### Check OpenSearch 

```bash
# View all papers and their status
python debugging/check_opensearch.py

# Clear OpenSearch (DO NOT RUN UNLESS ASKING DAN FIRST (anyways this file is gitignored))
python clear_opensearch.py
```

### Setup Queues (First Time Only)

```bash
# Create all SQS queues and configure Lambda triggers
chmod +x deployment/setup_sqs_queues.sh
./deployment/setup_sqs_queues.sh
```

### Initial Setup

```bash
# 1. Setup SQS queues and infrastructure
./deployment/setup_sqs_queues.sh

# 2. Setup pipeline (S3 buckets, IAM policies)
./deployment/setup_pipeline.sh

# 3. Deploy all Lambda functions
./deployment/deploy_all.sh

#4. Deploy indiviudal lambda
./deployment/build_judge.sh

# 4. Setup Trainium instance (if needed)
./deployment/deploy_trainium.sh /path/to/your-key.pem
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

CODE_TEST=$(aws sqs get-queue-url --queue-name code-testing.fifo --query 'QueueUrl' --output text)
aws sqs get-queue-attributes --queue-url $CODE_TEST \
  --attribute-names ApproximateNumberOfMessages \
  --query 'Attributes.ApproximateNumberOfMessages' --output text | xargs -I {} echo "code-testing: {} messages"
```

### Check OpenSearch Field Mapping

```bash
# View all fields in OpenSearch index (should show 66 fields)
python debugging/check_opensearch_mapping.py
```

---

## Environment Variables

### Scraper Lambda
- `CONFERENCE` - Conference to scrape ("ICLR", "ICML", "BOTH")
- `CONFERENCE_YEAR` - Year (default: "2025")
- `MAX_PAPERS` - Max papers to process (default: "3")
- `BUCKET_NAME` - S3 bucket for PDFs
- `QUEUE_URL` - SQS queue URL

### Judge Lambda
- `OPENSEARCH_ENDPOINT` - OpenSearch cluster endpoint
- `OPENSEARCH_INDEX` - Index name (default: "research-papers-v2")

### Code Generator Lambda
- `OPENSEARCH_ENDPOINT` - OpenSearch cluster endpoint
- `OPENSEARCH_INDEX` - Index name (default: "research-papers-v2")
- `BEDROCK_MODEL_ID` - Claude model ID
- `CODE_BUCKET` - S3 bucket for code (default: "papers-code-artifacts")
- `CODE_QUEUE_URL` - SQS queue URL for testing

### Code Tester Lambda
- `OPENSEARCH_ENDPOINT` - OpenSearch cluster endpoint
- `OPENSEARCH_INDEX` - Index name (default: "research-papers-v2")
- `OUTPUTS_BUCKET` - S3 bucket for test results (default: "papers-test-outputs")
- `TRAINIUM_ENDPOINT` - HTTP endpoint for Trainium instance (e.g., "http://10.0.1.50:8000")
- `TRAINIUM_INSTANCE_ID` - EC2 instance ID for auto-start (optional)
- `BATCH_SIZE` - Number of papers to batch together (default: 10)
- `TRAINIUM_TIMEOUT` - Execution timeout in seconds (default: 600)

---

## Trainium Instance Setup

### Launch Trainium Instance

```bash
# Launch trn1.2xlarge instance with Deep Learning AMI
aws ec2 run-instances \
  --image-id <DEEP_LEARNING_AMI_ID> \
  --instance-type trn1.2xlarge \
  --key-name <YOUR_KEY_NAME> \
  --security-group-ids <YOUR_SG_ID> \
  --subnet-id <YOUR_SUBNET_ID> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=PapersCodeTester-Trainium},{Key=Purpose,Value=PapersCodeTester}]'
```

### Deploy Trainium Executor (Ask Dan how to do this)

```bash
# Deploy Flask app to Trainium instance (from local machine)
./deployment/deploy_trainium.sh /path/to/your-key.pem

# OR setup on the Trainium instance directly (SSH into instance first)
./deployment/setup_trainium_remote.sh
```

### Trainium Instance Configuration

**Required `.env` variables:**
```bash
ASK DAN 
```

**To find our instance ID (can also just check console or ask Dan, but if needed...):**
```bash
aws ec2 describe-instances \
  --region us-east-2 \
  --filters "Name=instance-type,Values=trn1.2xlarge" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
  --output table

# via AWS Console: EC2 → Select your region → Instances → Find your Trainium instance → Copy Instance ID
```

### View Trainium Execution Logs (Real-time)

**Quick helper script** (automatically finds SSH key):
```bash
# Tail all logs (auto-detects SSH key from instance)
./tail_trainium_logs.sh

# Tail logs filtered for specific paper
./tail_trainium_logs.sh 6-j63JkBP8oloYi_8CJH
```

**What to look for in logs:**
- `"Executing code for paper <paper_id>"` - Execution started
- `"Paper <paper_id> executed successfully"` - Execution completed
- `"Logged metrics to CloudWatch"` - Metrics were sent
- `"Failed to"` or `"Error"` - Issues to investigate
- `METRICS:` lines - Training metrics being extracted

### SageMaker Metrics Tracking

Training metrics from Trainium executions are automatically logged to **CloudWatch Metrics** (SageMaker-compatible). This enables:

- Viewing training metrics in CloudWatch Console
- Visualizing metrics in SageMaker Studio
- Setting up CloudWatch alarms
- Tracking training progress across papers

**How it works:**
1. Generated code outputs metrics in format: `print(f"METRICS: {json.dumps({'training_loss': 0.023})}")`
2. Trainium executor automatically extracts and logs metrics to CloudWatch
3. Metrics are stored in namespace `Trainium/Training` with dimensions (PaperId, TrainingJobName, InstanceType)

**Viewing Metrics:**
- **CloudWatch Console**: Navigate to Metrics → Trainium/Training
- **SageMaker Studio**: Metrics appear in Experiments/Training Jobs dashboard
- **AWS CLI**: Use `aws cloudwatch list-metrics --namespace "Trainium/Training"`

**Configuration:**
- Enable/disable: Set `SAGEMAKER_METRICS_ENABLED=true` (default: enabled)
- IAM required: Trainium instance needs `cloudwatch:PutMetricData` permission

---

## Cost Estimate (per 100 papers)

- **Scraping**: ~$0.10 (Lambda + S3)
- **Judging**: ~$0.05 (Lambda + Claude API)
- **Code Generation**: ~$3.00 (Bedrock/Claude API calls)
- **Code Testing**: ~$1.30 (Trainium trn1.2xlarge @ $1.34/hr for ~1hr + Lambda dispatch)
- **Storage**: ~$0.01 (S3 + OpenSearch)

**Total**: ~$4.46 per 100 papers

**Note**: Trainium costs assume batching of 10 papers reduces total execution time. On-demand pricing: trn1.2xlarge = $1.34/hour.

```bash
# Stop instance (can be restarted later)
aws ec2 stop-instances --region us-east-2 --instance-ids i-0f0bf0de25aa4fd57

# Start instance when needed
aws ec2 start-instances --region us-east-2 --instance-ids i-0f0bf0de25aa4fd57

**Download Code/Results:**
```bash
# Download generated code from S3
python download_s3_code.py

# Download test results from S3
python download_test_results.py
```

### Setup Scripts

```bash
# Setup SQS queues and Lambda triggers (first time only)
./deployment/setup_sqs_queues.sh

# Setup pipeline infrastructure (S3 buckets, IAM policies) FIRST TIME ONLY
./deployment/setup_pipeline.sh

# Deploy Trainium executor (Ask Dan for SSH key)
./deployment/deploy_trainium.sh /path/to/your-key.pem

```

### Code gen --> test on trn
```bash
python grab_papers_for_code_gen.py ## this grabs 3 random papers from opensearch and then sends them to code eval SQS
python monitor_pipeline.py --paper-ids <Paper ID> <Paper ID> <Paper ID> --watch --interval 15
```

---

## Midpoint Deliverable Pipeline

Grabs a single random paper from opensearch, generates its code, sends it to the code reviewer, then the flask app executes it on trn & stores neuron profiler results + overall trn results.

### Quick Start

```bash
# Process a specific paper
python pipeline_for_delivery.py --paper-id <paper_id>

# Process recent papers
python pipeline_for_delivery.py --recent-days 30 --max-papers 5

# Check results for a paper
python debugging/check_error.py <paper_id>
```

### Key Features

- **Neuron SDK Integration**: Generated code uses `torch_xla` for Trainium compatibility
- **Hardware-Level Profiling**: Neuron Profiler captures hardware execution traces
- **CloudWatch Metrics**: Automatic logging of training and execution metrics

### Directory Structure

```
├── code_gen/                # Code generation module
├── trn_execute/             # Trainium execution module
├── results/                 # Results (per-paper folders)
├── debugging/               # Debugging and utility scripts
├── pipeline_for_delivery.py # Main pipeline script
└── storage_utils.py         # Storage utilities
```

---
