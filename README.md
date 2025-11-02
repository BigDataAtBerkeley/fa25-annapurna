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
3. **PaperScraper_ArXiv** - Scrapes ArXiv papers
4. **PapersJudge** - Evaluates paper relevance
5. **PapersCodeGenerator** - Generates PyTorch code from papers
6. **PapersCodeTester** - Batches code (10 at a time) and dispatches to Trainium instance for execution

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

### Deploy All Functions
```bash
chmod +x deployment/*.sh
./deployment/deploy_all.sh
```

### Deploy Individual Components
```bash
# Scrapers
./deployment/build_scraper.sh PaperScraper_ICLR
./deployment/build_scraper.sh PaperScraper_ICML

# Judge
./deployment/build_judge.sh

# Code Generator
./deployment/build_code_gen_lambda.sh

# Code Tester
./deployment/build_test_lambda.sh
```

---

## Testing & Usage

### Trigger Scrapers Manually

```bash


**"ICLR" IS USED AS AN EXAMPLE. TO TEST OTHER LAMBDA FUNCTIONS, REPLACE "ICLR" WITH "ICML", "arxiv", "MLSYS", or "NEURIPS".**


# Scrape ICLR papers
aws lambda invoke \
  --function-name PaperScraper_ICLR \
  --payload '{"MAX_PAPERS": "5"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

# Scrape ICML papers by topic
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "ICML", "topic_filter": "Deep Learning->Large Language Models"}' \
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
- Dataset loader deployed on Trainium (`./deployment/deploy_trainium.sh`)
- TRAINIUM_ENDPOINT and TRAINIUM_INSTANCE_ID in .env

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

2. Judge logs
aws logs tail /aws/lambda/PapersJudge --since 15m --follow

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
python check_opensearch.py

# Clear OpenSearch (DO NOT RUN UNLESS ASKING DAN FIRST (anyways this file is gitignored))
python clear_opensearch.py


### Setup Queues (First Time Only)

```bash
# Create all SQS queues and configure Lambda triggers
chmod +x deployment/setup_sqs_queues.sh
./deployment/setup_sqs_queues.sh
```

python check_opensearch.py
```


```bash
# Get queue URL
CODE_EVAL_QUEUE=$(aws sqs get-queue-url --queue-name code-evaluation.fifo --query 'QueueUrl' --output text)


# Watch code generation logs
aws logs tail /aws/lambda/PapersCodeGenerator --follow
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
python check_opensearch_mapping.py
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
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=PapersCodeTester-Trainium}]'
```
---

## Cost Estimate (per 100 papers)

- **Scraping**: ~$0.10 (Lambda + S3)
- **Judging**: ~$0.05 (Lambda + Claude API)
- **Code Generation**: ~$3.00 (Bedrock/Claude API calls)
- **Code Testing**: ~$1.30 (Trainium trn1.2xlarge @ $1.34/hr for ~1hr + Lambda dispatch)
- **Storage**: ~$0.01 (S3 + OpenSearch)

**Total**: ~$4.46 per 100 papers

**Note**: Trainium costs assume batching of 10 papers reduces total execution time. On-demand pricing: trn1.2xlarge = $1.34/hour.

---
