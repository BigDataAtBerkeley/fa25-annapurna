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

# Judge logs
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
```

### Check Trainium Instance

```bash
# Check instance status
aws ec2 describe-instances \
  --instance-ids <TRAINIUM_INSTANCE_ID> \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text

# Start Trainium instance manually (if needed)
aws ec2 start-instances --instance-ids <TRAINIUM_INSTANCE_ID>

# Stop Trainium instance (to save costs when not in use)
aws ec2 stop-instances --instance-ids <TRAINIUM_INSTANCE_ID>

# SSH into Trainium instance (for debugging)
ssh -i <your-key.pem> ubuntu@<TRAINIUM_PUBLIC_IP>

# Check Trainium service logs
ssh ubuntu@<TRAINIUM_PUBLIC_IP> "tail -f /var/log/trainium-executor.log"
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

### Prerequisites
- AWS account with access to Trainium instances (trn1 family)
- VPC with subnet and security group configured
- SSH key pair for instance access

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

### Setup Trainium Executor Service

SSH into the instance and run:

```bash
# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv

# Install PyTorch Neuron
pip3 install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com

# Create executor service directory
mkdir -p ~/trainium-executor
cd ~/trainium-executor

# Create Python service (see trainium_executor.py below)
# This service should expose an HTTP endpoint at port 8000
# Endpoint: POST /execute_batch
# Accepts: {"batch": [{"paper_id": "...", "code": "..."}]}
# Returns: {"results": {"paper_id": {"success": bool, "stdout": "...", ...}}}

# Run as systemd service (recommended for production)
sudo systemctl enable trainium-executor
sudo systemctl start trainium-executor
```

### Configure Security Group

Ensure the security group allows:
- Inbound TCP 8000 from Lambda VPC (for HTTP API)
- Inbound TCP 22 from your IP (for SSH access)

### Lambda Configuration

- Deploy Lambda in same VPC as Trainium instance
- Set `TRAINIUM_ENDPOINT` to private IP of instance (e.g., `http://10.0.1.50:8000`)
- Configure SQS trigger with:
  - **BatchSize**: 10
  - **MaximumBatchingWindowInSeconds**: 60 (wait up to 60s to accumulate 10 messages)

---

## 🎯 Complete E2E Example

```bash
# 1. Scrape papers
aws lambda invoke --function-name PaperScraper_ICLR \
  --payload '{"MAX_PAPERS": "10"}' response.json

# 2. Wait for judge to process (automatic via SQS)
# Papers are automatically sent to OpenSearch

# 3. Generate code for accepted papers
aws lambda invoke --function-name PapersCodeGenerator \
  --payload '{"action":"generate_recent","max_papers":10}' response.json

# 4. Wait for code testing (automatic via SQS + Trainium)
# - Papers accumulate in SQS (up to 10)
# - Lambda batches them and sends to Trainium
# - Test results automatically saved to OpenSearch

# 5. Check results in OpenSearch
python check_opensearch.py
```

## 💰 Cost Estimate (per 100 papers)

- **Scraping**: ~$0.10 (Lambda + S3)
- **Judging**: ~$0.05 (Lambda + Claude API)
- **Code Generation**: ~$3.00 (Bedrock/Claude API calls)
- **Code Testing**: ~$1.30 (Trainium trn1.2xlarge @ $1.34/hr for ~1hr + Lambda dispatch)
- **Storage**: ~$0.01 (S3 + OpenSearch)

**Total**: ~$4.46 per 100 papers

**Note**: Trainium costs assume batching of 10 papers reduces total execution time. On-demand pricing: trn1.2xlarge = $1.34/hour.

---
