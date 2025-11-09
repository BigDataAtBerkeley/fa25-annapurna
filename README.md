# Annapurna - Research Paper Pipeline

Automated pipeline for scraping, evaluating, generating code, and testing ML research papers on AWS Trainium.

---

## Architecture Overview

```
Paper Scrapers (Lambda functions for ICLR, ICML, arXiv, NEURIPS, MLSys)
    ↓
S3 (llm-research-papers) --> all raw papers stored here (literally every single paper scraped. no papers ever deleted from this bucket)
    ↓
SQS (researchQueue.fifo) --> every paper from S3 thats initially scraped gets sent here. should be empty 99% of the time bc papersJudge Lambda is invoked every time a paper hits the front of this queue
    ↓
PapersJudge Lambda (filters relevant papers) --> every paper in researchQueue.fifo gets sent here, where a call to Claude via Bedrock determines if it should be indexed in OpenSearch (based on relevance and novelty)
    ↓
OpenSearch (research-papers-v2) → Indexed papers with metadata
    ↓
SQS (code-evaluation.fifo) → Papers pending code generation (batches of 10)
    ↓
PapersCodeGenerator Lambda → Generates PyTorch code using Claude via Bedrock
    ↓
S3 (papers-code-artifacts) + OpenSearch → Generated code stored in S3, metadata in OpenSearch
    ↓
SQS (code-testing.fifo) → Code files pending testing (batches of 10)
    ↓
PapersCodeTester Lambda → Downloads code, batches 10 papers, sends to Trainium
    ↓
Trainium Instance (trn1.2xlarge) → Executes PyTorch code with AWS Neuron SDK
    ↓
S3 (papers-test-outputs) + OpenSearch → Test results stored in S3, attached to papers in OpenSearch
```

---

## Components

### Lambda Functions

1. **PaperScraper_ICLR** - Scrapes ICLR conference papers
2. **PaperScraper_ICML** - Scrapes ICML conference papers
3. **PaperScraper_arxiv** - Scrapes ArXiv papers
4. **PaperScraper_NEURIPS** - Scrapes NeurIPS conference papers
5. **PaperScraper_MLSYS** - Scrapes MLSys conference papers
6. **PapersJudge** - Evaluates paper relevance and novelty using Claude
7. **PapersCodeGenerator** - Generates PyTorch code from papers using Claude
8. **PapersCodeTester** - Batches code (10 at a time) and dispatches to Trainium
9. **LogCleanupLambda** - Cleans up Lambda logs

### Compute Resources

- **Trainium Instance (trn1.2xlarge)** 
  - Auto-starts when batch is ready
  - Auto-stops when queue is empty
  - Flask app running on port 8000 for code execution

### S3 Buckets

- `llm-research-papers` - Scraped papers (PDFs, never deleted)
- `papers-code-artifacts` - Generated PyTorch code files
- `papers-test-outputs` - Code execution logs, stdout, stderr, plots, metrics
- `datasets-for-all-papers` - Standardized datasets for testing (CIFAR-10, MNIST, IMDB, etc.)

### SQS Queues

- `researchQueue.fifo` - Papers pending evaluation (processed one at a time)
- `code-evaluation.fifo` - Papers pending code generation (batches of 10 or 24 hours)
- `code-testing.fifo` - Code files pending testing (batches of 10 or 1 hour)

### OpenSearch Index

- `research-papers-v2` - All papers with:
  - Metadata (title, authors, abstract, date)
  - Code status (generated, S3 location)
  - Test results (success, execution time, metrics, S3 artifacts)
  - Dataset information

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

# 3. Setup datasets in S3 (DAN ALRDY DID THIS, DONT RUN AGAIN --> anyways gitignored)
cd datasets
python upload_datasets_to_s3.py

# 4. Deploy Trainium executor (ask Dan for ssh key )
./deployment/deploy_trainium.sh /path/to/your-key.pem
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

## Usage

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

# Generate code for recent papers
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --payload '{"action":"generate_recent","max_papers":5}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

**Available Actions:**
- `generate_by_id` - Generate code for a specific paper by ID
- `generate_by_ids` - Generate code for multiple papers by IDs
- `generate_by_title` - Generate code for papers matching a title
- `generate_by_author` - Generate code for papers by a specific author
- `generate_by_keywords` - Generate code for papers matching abstract keywords
- `generate_recent` - Generate code for recently ingested papers
- `get_paper_info` - Get paper information without generating code

### Download Generated Code Locally

```bash
# Download code files from S3
python download_s3_code.py

# This will:
# 1. List all available code files in S3
# 2. Let you choose which ones to download
# 3. Save to generated_code/ directory with metadata
```

### Download Test Results

```bash
# Download test results from S3
python download_test_results.py

# This will:
# 1. List all papers with test results
# 2. Let you choose which results to download
# 3. Save stdout.log, stderr.log, metrics.json, and plots to test_results/
```

### CLI Usage (Local Code Generation (THIS DOESNT SEND ANYTHING TO AWS, JUST SAVES OUTPUTS LOCALLY))

```bash
# Generate code by paper ID (local, no AWS)
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

## Datasets

### Available Datasets

All datasets are stored in S3 bucket `datasets-for-all-papers` and automatically downloaded to Trainium instances when needed

| Dataset | Type | Size | Samples | Use Case |
|---------|------|------|---------|----------|
| **CIFAR-10** | Image Classification | ~170 MB | 60K | Computer vision models, CNNs |
| **CIFAR-100** | Image Classification | ~170 MB | 60K | Multi-class vision tasks |
| **MNIST** | Image Classification | ~12 MB | 70K | Simple digit classification |
| **Fashion-MNIST** | Image Classification | ~30 MB | 70K | Fashion item classification |
| **IMDB** | Text Classification | ~65 MB | 50K | Sentiment analysis, NLP |
| **WikiText-2** | Language Modeling | ~12 MB | 36K | Language models, transformers |
| **Synthetic** | Various | ~500 MB | 16K | Quick testing, debugging |

### Setup Datasets

```bash
cd datasets

# Activate virtual environment
source ../aws_env/bin/activate

# Install dependencies
pip install boto3 torch torchvision datasets tqdm

# Upload all datasets to S3
python upload_datasets_to_s3.py
```

### Using Datasets in Generated Code

Generated code should use the standardized dataset loader:

```python
from dataset_loader import load_dataset

# For vision tasks, use CIFAR-10:
train_loader, test_loader = load_dataset('cifar10', batch_size=128)

# For NLP tasks, use IMDB:
train_data, test_data = load_dataset('imdb')

# For quick testing, use synthetic data:
data = load_dataset('synthetic', variant='small')
```

**Available datasets:** `cifar10`, `cifar100`, `mnist`, `fashion_mnist`, `imdb`, `wikitext2`, `synthetic`

### Dataset Caching

Datasets are cached on Trainium instances at `/tmp/datasets`, so download costs are one-time per instance lifecycle. The dataset loader automatically:
1. Checks local cache first
2. Downloads from S3 if not cached
3. Returns PyTorch DataLoaders ready for training

---

## Trainium Instance Setup

### Launch Trainium Instance

```bash
aws ec2 run-instances \
  --image-id <DEEP_LEARNING_AMI_ID> \
  --instance-type trn1.2xlarge \
  --key-name <YOUR_KEY_NAME> \
  --security-group-ids <YOUR_SG_ID> \
  --subnet-id <YOUR_SUBNET_ID> \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=PapersCodeTester-Trainium},{Key=Purpose,Value=PapersCodeTester}]'
```

### Deploy Trainium Executor

```bash
# Deploy Flask app to Trainium instance (from local) --> ask Dan for ssh key
./deployment/deploy_trainium.sh /path/to/your-key.pem

# OR setup on the Trainium instance directly (SSH into instance first) --> ask Dan for ssh key
./deployment/setup_trainium_remote.sh
```

### Trainium Instance Management

The Trainium instance automatically:
- **Starts** when a batch of 10 papers is ready in `code-testing.fifo`
- **Stops** after processing batch if queue is empty (saves costs)

**Manual control:**
```bash
# Stop instance (can be restarted later)
aws ec2 stop-instances --region us-east-2 --instance-ids i-0f0bf0de25aa4fd57

# Start instance when needed
aws ec2 start-instances --region us-east-2 --instance-ids i-0f0bf0de25aa4fd57

# Terminate instance (permanent - cannot be restarted)
aws ec2 terminate-instances --region us-east-2 --instance-ids i-0f0bf0de25aa4fd57
```

**Note:** We use Elastic IP to be able to start & stop trn --> this incurs charge of ~$3.60/mth even when the instance is stopped

### Finding Our Trainium Instance

```bash
aws ec2 describe-instances \
  --region us-east-2 \
  --filters "Name=instance-type,Values=trn1.2xlarge" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
  --output table
```

---

## Monitoring

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

# Code Tester logs
aws logs tail /aws/lambda/PapersCodeTester --since 5m --follow
```

### Check SQS Queues

```bash
# Papers queue (scraper → judge)
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo \
  --attribute-names ApproximateNumberOfMessages

# Code evaluation queue (judge → code generator)
CODE_EVAL=$(aws sqs get-queue-url --queue-name code-evaluation.fifo --query 'QueueUrl' --output text)
aws sqs get-queue-attributes --queue-url $CODE_EVAL \
  --attribute-names ApproximateNumberOfMessages \
  --query 'Attributes.ApproximateNumberOfMessages' --output text

# Code testing queue (code generator → tester)
CODE_TEST=$(aws sqs get-queue-url --queue-name code-testing.fifo --query 'QueueUrl' --output text)
aws sqs get-queue-attributes --queue-url $CODE_TEST \
  --attribute-names ApproximateNumberOfMessages \
  --query 'Attributes.ApproximateNumberOfMessages' --output text
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

### Check OpenSearch

```bash
# View all papers and their status
python check_opensearch.py

# View OpenSearch field mapping (keys/fields)
python check_opensearch_mapping.py

# Clear OpenSearch (DO NOT RUN UNLESS ASKING DAN FIRST)
# python clear_opensearch.py
```

### View Trainium Execution Logs

SSH into your Trainium instance (ask Dan for ssh key):

```bash
ssh -i your-key.pem ubuntu@<TRAINIUM_IP>

# View Flask app logs
tail -f ~/trainium-executor/logs/trainium-executor.log

# View systemd service logs
sudo journalctl -u trainium-executor.service -f
```

**What to look for in logs:**
- `"Executing code for paper <paper_id>"` - Execution started
- `"Paper <paper_id> executed successfully"` - Execution completed
- `"Logged metrics to CloudWatch"` - Metrics were sent
- `"Failed to"` or `"Error"` - Issues to investigate
- `METRICS:` lines - Training metrics being extracted

### SageMaker Metrics Tracking

Training metrics from Trainium executions are automatically logged to **CloudWatch Metrics** (SageMaker-compatible).

**How it works:**
1. Generated code outputs metrics in format: `print(f"METRICS: {json.dumps({'training_loss': 0.023})}")`
2. Trainium executor automatically extracts and logs metrics to CloudWatch
3. Metrics are stored in namespace `Trainium/Training` with dimensions (PaperId, TrainingJobName, InstanceType)

**Viewing Metrics:**
- **CloudWatch Console**: Navigate to Metrics → Trainium/Training
- **AWS CLI**: 
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

**Configuration:**
- Enable/disable: Set `SAGEMAKER_METRICS_ENABLED=true` (default: enabled)
- IAM required: Trainium instance needs `cloudwatch:PutMetricData` permission

---

## Configuration

### Environment Variables

#### Scraper Lambda
- `CONFERENCE` - Conference to scrape ("ICLR", "ICML", "BOTH")
- `CONFERENCE_YEAR` - Year (default: "2025")
- `MAX_PAPERS` - Max papers to process (default: "3")
- `BUCKET_NAME` - S3 bucket for PDFs
- `QUEUE_URL` - SQS queue URL

#### Judge Lambda
- `OPENSEARCH_ENDPOINT` - OpenSearch cluster endpoint
- `OPENSEARCH_INDEX` - Index name (default: "research-papers-v2")
- `CODE_EVAL_QUEUE_URL` - SQS queue for code generation (code-evaluation.fifo)

#### Code Generator Lambda
- `OPENSEARCH_ENDPOINT` - OpenSearch cluster endpoint
- `OPENSEARCH_INDEX` - Index name (default: "research-papers-v2")
- `BEDROCK_MODEL_ID` - Claude model ID
- `CODE_BUCKET` - S3 bucket for code (default: "papers-code-artifacts")
- `CODE_TEST_QUEUE_URL` - SQS queue for testing (code-testing.fifo)

#### Code Tester Lambda
- `OPENSEARCH_ENDPOINT` - OpenSearch cluster endpoint
- `OPENSEARCH_INDEX` - Index name (default: "research-papers-v2")
- `OUTPUTS_BUCKET` - S3 bucket for test results (default: "papers-test-outputs")
- `TRAINIUM_ENDPOINT` - HTTP endpoint for Trainium instance (e.g., "http://10.0.1.50:8000")
- `TRAINIUM_INSTANCE_ID` - EC2 instance ID for auto-start (optional)
- `BATCH_SIZE` - Number of papers to batch together (default: 10)
- `TRAINIUM_TIMEOUT` - Execution timeout in seconds (default: 1800 = 30 minutes)
- `CODE_TEST_QUEUE_URL` - SQS queue URL (code-testing.fifo)

#### Trainium Executor
- `MAX_EXECUTION_TIME` - Maximum execution time per code file (default: 1800 seconds)
- `DATASET_CACHE_DIR` - Directory for caching datasets (default: "/tmp/datasets")
- `SAGEMAKER_METRICS_ENABLED` - Enable CloudWatch metrics logging (default: "true")

---

## Cost Estimates

### Per 100 Papers

- **Scraping**: ~$0.10 (Lambda + S3)
- **Judging**: ~$0.05 (Lambda + Claude API)
- **Code Generation**: ~$3.00 (Bedrock/Claude API calls)
- **Code Testing**: ~$1.30 (Trainium trn1.2xlarge @ $1.34/hr for ~1hr + Lambda dispatch)
- **Storage**: ~$0.01 (S3 + OpenSearch)
- **Elastic IP** (if instance stopped): ~$0.30/month

**Total**: ~$4.46 per 100 papers

**Note**: Trainium costs assume batching of 10 papers reduces total execution time. On-demand pricing: trn1.2xlarge = $1.34/hour.

### Dataset Storage Costs

| Dataset | S3 Storage Cost/Month | Download Cost (per execution) |
|---------|----------------------|-------------------------------|
| CIFAR-10 | ~$0.004 | ~$0.017 (first time only, then cached) |
| IMDB | ~$0.002 | ~$0.007 (first time only, then cached) |
| WikiText-2 | ~$0.0003 | ~$0.001 (first time only, then cached) |
| **Total** | **~$0.025/month** | **Cache on Trainium = $0 after first download** |

---

## Troubleshooting

### Pipeline Issues

**Issue: Papers not being processed**
- Check SQS queue depths (see Monitoring section)
- Verify Lambda functions are deployed and have correct IAM permissions
- Check CloudWatch logs for errors

**Issue: Code not being generated**
- Verify `CODE_EVAL_QUEUE_URL` is set in Judge Lambda
- Check if paper already has code generated (Judge Lambda skips these)
- Check PapersCodeGenerator Lambda logs

**Issue: Code not being tested**
- Verify `CODE_TEST_QUEUE_URL` is set in Code Generator Lambda
- Check if 10 papers have accumulated in `code-testing.fifo`
- Verify Trainium instance is running and accessible

### Trainium Issues

**Issue: Connection errors**
- Check `TRAINIUM_ENDPOINT` is correct
- Verify Trainium Flask app is running: `curl http://your-endpoint:8000/health`
- Check security groups allow access from Lambda VPC

**Issue: Instance not starting**
- Verify `TRAINIUM_INSTANCE_ID` is correct
- Check AWS credentials have EC2 permissions
- Verify instance exists in the specified region

**Issue: Execution timeouts**
- Increase `TRAINIUM_TIMEOUT` (default: 1800 seconds = 30 minutes)
- Check Trainium instance has enough resources
- Verify code doesn't have infinite loops
- Neuron compilation can take 5-15 minutes for large models

**Issue: Dataset not found**
- Verify bucket exists: `aws s3 ls s3://datasets-for-all-papers/`
- Check IAM permissions on Trainium instance
- Re-run upload script: `python datasets/upload_datasets_to_s3.py`

**Issue: Dataset loader import error**
- Ensure `dataset_loader.py` is in the Trainium executor directory
- Redeploy using `./deployment/deploy_trainium.sh`

### Code Generation Issues

**Issue: Generated code has errors**
- Check `code_gen/bedrock_client.py` prompt for latest guidance
- Verify dataset recommendations are correct
- Check generated code uses `dataset_loader` correctly

**Issue: Missing imports in generated code**
- The code generation prompt includes explicit import instructions
- Check Bedrock model version and capabilities

---

## Code Generation Details

### How Code Generation Works

1. **Paper Retrieval**: Fetches paper from OpenSearch or S3
2. **Dataset Recommendation**: Analyzes paper to recommend appropriate dataset
3. **Code Generation**: Uses Claude via AWS Bedrock to generate PyTorch code
4. **Code Validation**: Checks for required imports and dataset usage
5. **S3 Storage**: Saves code to `papers-code-artifacts` bucket
6. **OpenSearch Update**: Updates paper document with code metadata
7. **Queue Dispatch**: Sends paper to `code-testing.fifo` for execution

### Generated Code Requirements

Generated code must:
- Use `from dataset_loader import load_dataset` for data loading
- Include proper imports (torch, torch.nn, torch.optim, etc.)
- Follow PyTorch Transformer API correctly (no `mask=` keyword for TransformerEncoder/Decoder)
- Avoid in-place operations on XLA tensors (use `torch.clamp()` instead of `tensor[tensor == 0] = 1`)
- Avoid `torch.dot()` on batched tensors (use `torch.bmm()` or `torch.einsum()`)
- Use scalar tensors for SLERP interpolation parameters
- Output metrics in format: `print(f"METRICS: {json.dumps({'training_loss': 0.023})}")`

### Dataset Integration

The code generation system automatically:
- Recommends appropriate datasets based on paper content
- Includes dataset loading code in generated files
- Ensures compatibility with Trainium execution environment

---

## Additional Resources

### Local Testing Utilities

**Test Code Generation (without AWS):**
```bash
# Test code generation locally
python code-gen-testing/test_code_generation.py

# Sample random papers and generate code
python code-gen-testing/random_sample_generate.py --count 5
```

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

---

