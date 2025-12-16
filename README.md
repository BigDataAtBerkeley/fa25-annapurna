# Annapurna - Research Paper Pipeline

Automated pipeline for scraping, evaluating, and generating code from ML research papers.

---

## Architecture Overview
![Alt text](./Diagram.png)
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
1. **PaperScraperConferences** - Scrapes ICLR, ICML, NeurIPS, MLSYS papers
2. **conferenceWrapper** - Wrapper function to enable MapState for PaperScraperConferences.
3. **PaperScraper_arxiv** - Scrapes ArXiv papers
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
Go into the setup folder and follow the README.md instructions. 

### Deploy all Functions (After Initial Setup)
```bash
# Scrapers
./deployment/build_scraper.sh PaperScraperConferences
./deployment/build_scraper.sh PaperScraper_arxiv
./deployment/build_conference_wrapper.sh

# Judge
./deployment/build_judge.sh

# Code Generator
# Train page classifier model
python code_gen/page_classifier.py --train-csv data/synthetic_labeled_data.csv
# Make sure this model is in the code_gen folder. 
./deployment/build_code_gen_lambda.sh

# Code Tester
./deployment/build_test_lambda.sh

# Cleanup Lambda (optional)
./deployment/build_cleanup.sh
```

---

## Testing & Usage

When editing the prompt for judge_lambda, we reccomend first edtiing judge_lambda_test and then running 

```bash
python debugging/judge_lambda_test.py
```
This will randomly sample papers already in OpenSearch and test against how your new classifier evaluates them. 
(There is an anti-self comparison mechanism in the test not included in judge_lambda).

```bash
python check_opensearch.py 
```
This returns all the papers that have been scraped. 

The remaining files you are free to explore and have descriptions for the use case in each, but these two are the most frequently referenced. 

### Trigger Scrapers Manually

```bash

# Execute Scraping of Conference Papers via MapState. This should be used in production.
# If it's failing above 10 batches, check concurenncy limit. It's currently set to 100 but can be maxed at 1000.
aws stepfunctions start-execution \
  --state-machine-arn [INSERT THE STATE MACHINE ARN] \
  --name "test-60-papers-$(date +%s)" \
  --input '{"source": "iclr", "year": 2025, "search_term": "LLM", "batch_size": 30, "test_count": 300}'

# If you want to scrape everything, just don't add the test_count value.

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

# For ArXiv
aws lambda invoke \
  --function-name PaperScraper_arxiv \
  --payload '{"MAX_PAPERS": "5"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json

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
aws logs tail /aws/lambda/PaperScraperConferences --since 5m --follow
aws logs tail /aws/lambda/PaperScraper_arxiv --since 5m --follow

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
- `SOURCE` - Conference to scrape ("ICLR", "ICML", "BOTH")
- `SCRAPER_YEAR` - Year (default: "2025")
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

### Viewing Profiler Trace Files in Perfetto

When code executes successfully on Trainium, profiler artifacts (trace files) are automatically collected and uploaded to S3. These can be viewed in the Perfetto UI for performance analysis.

**Location of Trace Files:**
- **S3**: `s3://trainium-execution-results/profiler/{paper_id}/`
- **Local (temporary)**: `/tmp/neuron_profiler/{paper_id}_{timestamp}/` (may be cleaned up)

**Files Generated:**
- `ntrace.pb` - Main neuron trace file (large, ~100MB+)
- `cpu_util.pb` - CPU utilization trace
- `host_mem.pb` - Host memory trace
- `trace_info.pb` - Trace metadata
- `system_profile.pftrace` - Converted Perfetto format (created after conversion)

**Step 1: Download Trace Files from S3**

```bash
# Download all profiler artifacts for a paper
aws s3 sync s3://trainium-execution-results/profiler/{paper_id}/ ~/Downloads/trace_files_{paper_id}/

# Or download just the main trace file
aws s3 cp s3://trainium-execution-results/profiler/{paper_id}/ntrace.pb ~/Downloads/ntrace.pb
```

**Step 2: Convert .pb Files to Perfetto Format (if needed)**

If you only have `.pb` files, convert them to `.pftrace` format for Perfetto:

```bash
# SSH into Trainium instance
ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@<TRAINIUM_IP>

# Download trace files to Trainium
aws s3 sync s3://trainium-execution-results/profiler/{paper_id}/ ~/trace_files_{paper_id}/

# Convert to Perfetto format (outputs to current directory)
cd ~/trace_files_{paper_id}
/opt/aws/neuron/bin/neuron-profile view -d . --output-format=perfetto

# Check if .pftrace file was created (usually named system_profile.pftrace)
ls -lah ~/*.pftrace

# Upload converted file back to S3
aws s3 cp ~/system_profile.pftrace s3://trainium-execution-results/profiler/{paper_id}/system_profile.pftrace

# Download to local machine
# From your local machine:
aws s3 cp s3://trainium-execution-results/profiler/{paper_id}/system_profile.pftrace ~/Downloads/system_profile.pftrace
```

**Step 3: View in Perfetto UI**

1. Go to https://ui.perfetto.dev/
2. Click "Open trace file" or drag and drop
3. Upload the `.pftrace` file 
4. Wait for upload and processing (large files may take a minute)

**What You'll See in Perfetto:**
- Timeline of execution showing all operations
- Neuron device operations and compute kernels
- CPU and memory usage over time
- Performance bottlenecks and idle time
- Detailed execution flow and dependencies

**Note:** The conversion command (`neuron-profile view`) creates `.pftrace` files in the **current working directory**, not in the input directory. After conversion, check your home directory (`~`) for `system_profile.pftrace` and `device_profile_model_*.pftrace` files.

**Accessing from Execution Results JSON:**
The execution results JSON (at `s3://trainium-execution-results/results/{paper_id}/execution_result.json`) includes:
- `profiler_s3_location`: S3 path to profiler artifacts
- `profiler_s3_console_url`: Direct link to view files in S3 console

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

## Slack Bot Setup

### Step 1: Create a Slack App

1. Go to https://api.slack.com/apps
2. Click **"Create New App"** → **"From scratch"**
3. Enter app name (e.g., "Annapurna Pipeline") and select your workspace
4. Click **"Create App"**

### Step 2: Configure Bot Token Scopes

1. In your app settings, go to **"OAuth & Permissions"** (left sidebar)
2. Scroll to **"Scopes"** → **"Bot Token Scopes"**
3. Add the following scopes:
   - `chat:write` - Send messages to channels
   - `chat:write.public` - Send messages to public channels (if needed)
   - `channels:read` - View basic information about public channels (optional)
4. Scroll up and click **"Install to Workspace"**
5. Authorize the app and copy the **"Bot User OAuth Token"** (starts with `xoxb-`)

### Step 3: Add Bot to Channel

1. In Slack, go to the channel where you want notifications
2. Type `/invite @YourBotName` or add the bot through channel settings
3. Note the channel name or ID (e.g., `#research-papers` or `C1234567890`)

### Step 4: Configure Environment Variables

Set these environment variables in your Lambda functions (should be auto-created after setup):

**For PapersCodeGenerator-container Lambda:**
```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_CHANNEL=your-channel-name-or-id
```

**Update Lambda environment variables:**
```bash
aws lambda update-function-configuration \
  --function-name PapersCodeGenerator-container \
  --environment "Variables={...,SLACK_BOT_TOKEN=xoxb-your-token,SLACK_CHANNEL=your-channel}" \
  --region us-east-1
```

**For Trainium executor (if using):**
Add to your `.env` file or environment:
```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_CHANNEL=your-channel-name-or-id
```

### What Gets Notified

The Slack bot sends notifications for:
- **New papers processed** - Initial paper information (title, authors, abstract)
- **Code generation** - When code is generated for a paper (with code length, model used, dataset)
- **Code execution** - When code is executed on Trainium (success/failure, execution time)
- **Execution results** - Training metrics and execution outcomes

All notifications for the same paper are threaded together for easy tracking.

### Testing

To test if the Slack bot is configured correctly:
```bash
# Test from PapersCodeGenerator Lambda
aws lambda invoke \
  --function-name PapersCodeGenerator-container \
  --payload '{"action":"generate_by_id","paper_id":"test-paper-id"}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

Check your Slack channel for notifications. If you don't see messages, verify:
- Bot token is correct (starts with `xoxb-`)
- Bot is added to the channel
- Channel name/ID is correct
- Lambda environment variables are set correctly

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
