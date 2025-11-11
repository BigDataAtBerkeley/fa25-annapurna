# Midpoint Deliverable - Research Paper Pipeline

Randomly grabs papers from opensearch, generates their code, sends to code reviewer, and execution on AWS Trainium with Neuron SDK and hardware-level profiling.

## Directory Structure

```
midpoint-deliverable/
├── code-gen-for-deliv/          # Code generation module
│   ├── bedrock_client.py        # Bedrock client with Neuron SDK prompts
│   ├── code_review_agent.py     # Code review with Neuron SDK checks
│   ├── pytorch_generator.py     # Main code generation orchestrator
│   ├── opensearch_client.py    # OpenSearch client for paper retrieval
│   ├── dataset_recommender.py  # Dataset recommendation logic
│   └── requirements.txt         # Dependencies
│
├── trn-execute-for-deliv/       # Trainium execution module
│   ├── app.py                   # Flask app with execute_code function
│   ├── dataset_loader.py        # Dataset loading utilities
│   ├── sagemaker_metrics.py     # CloudWatch metrics logging
│   └── requirements.txt         # Dependencies
│
├── results/                     # Results directory (per-paper structure)
│   └── {paper_id}/              # Each paper gets its own folder
│       ├── code-generation/     # Generated code and metadata
│       ├── code-review/         # Reviewed code and fixes
│       ├── trn-execution/       # Execution results
│       └── profiler_results/    # Neuron Profiler results (if available)
│
├── profiler_results/            # Downloaded profiler files (Perfetto traces, etc.)
│
├── pipeline_for_delivery.py    # Main pipeline script
├── check_results.py             # Utility to check results for a paper
└── view_profiler_results.py     # Utility to view profiler results
```

## Pipeline Flow

The pipeline processes one paper at a time through these steps:

1. **Paper Retrieval** → Fetch paper from OpenSearch (or use provided paper ID)
2. **Code Generation** → Generate PyTorch code using Bedrock (with Neuron SDK requirements)
3. **Code Review** → Review and fix code to ensure Neuron SDK compatibility
4. **Trainium Instance Management** → Start EC2 instance if stopped, wait for services
5. **TRN Execution** → Execute code on Trainium (with Neuron Profiler)
6. **Metrics Collection** → Extract and save metrics (CloudWatch + local)
7. **Profiler Results** → Save Neuron Profiler results

## Setup

### 1. Install Dependencies

```bash
# Code generation dependencies
cd midpoint-deliverable/code-gen-for-deliv
pip install -r requirements.txt

# Trainium execution dependencies (on Trainium instance)
cd ../trn-execute-for-deliv
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root or set environment variables:

#### Local Pipeline (pipeline_for_delivery.py)

```bash
# OpenSearch
export OPENSEARCH_ENDPOINT="your-opensearch-endpoint"
export OPENSEARCH_INDEX="research-papers-v2"
export AWS_REGION="us-east-1"

# Bedrock (using Claude 3 Sonnet for better rate limits)
export BEDROCK_MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0"

# Trainium EC2 Instance (for automatic instance management)
export TRAINIUM_INSTANCE_ID="i-xxxxxxxxxxxxx"  # EC2 instance ID
export TRAINIUM_ENDPOINT="http://1.2.3.4:8000"  # Trainium executor endpoint
export TRAINIUM_REGION="us-east-2"  # Region where Trainium instance is located
export TRAINIUM_EXECUTION_TIMEOUT="3600"  # Execution timeout in seconds

# SageMaker Metrics (optional, defaults to enabled)
export SAGEMAKER_METRICS_ENABLED="true"
```

#### Trainium Instance (trn-execute-for-deliv/app.py)

These variables are set **on the Trainium EC2 instance** (alrdy done by Dan):

```bash
# Neuron Profiler
export NEURON_PROFILER_ENABLED="true"
export PROFILER_OUTPUT_DIR="/tmp/neuron_profiler"

# Execution settings
export MAX_EXECUTION_TIME="1800"  # 30 minutes
export WORKING_DIR="/tmp/trainium_jobs"
export DATASET_CACHE_DIR="/tmp/datasets"
export SAGEMAKER_METRICS_ENABLED="true"
```

**To set on Trainium:** (alrdy done by Dan)
```bash
# SSH into Trainium
ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129

# Add to ~/.bashrc (persistent)
echo 'export NEURON_PROFILER_ENABLED="true"' >> ~/.bashrc
echo 'export PROFILER_OUTPUT_DIR="/tmp/neuron_profiler"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Deploy Trainium Executor

```bash
# Copy executor files to Trainium
scp -i ~/.ssh/trainium-deploy-key.pem -r midpoint-deliverable/trn-execute-for-deliv/* ec2-user@3.21.7.129:~/trainium-executor/

# SSH into Trainium and start executor (ask dan for key)
ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129
cd ~/trainium-executor
source venv/bin/activate  # Or create venv if needed
pip install -r requirements.txt
python app.py > logs/executor.log 2>&1 &
```

## Usage

### Process a Specific Paper

```bash
cd midpoint-deliverable
python pipeline_for_delivery.py --paper-id <paper_id>
```

### Process Recent Papers

```bash
python pipeline_for_delivery.py --recent-days 30 --max-papers 5
```

### Process Papers Matching a Query

```bash
python pipeline_for_delivery.py --query '{"match": {"title": "ResNet"}}' --max-papers 3
```

### Check Results for a Paper

```bash
python check_results.py <paper_id>
```

## Results Structure

Each paper gets its own folder in `results/`:

```
results/
└── {paper_id}/
    ├── code-generation/
    │   ├── {paper_id}_{timestamp}.py      # Generated code
    │   └── {paper_id}_{timestamp}.json    # Generation metadata
    │
    ├── code-review/
    │   ├── {paper_id}_{timestamp}.py      # Reviewed code
    │   └── {paper_id}_{timestamp}.json    # Review metadata
    │
    ├── trn-execution/
    │   └── {paper_id}_{timestamp}.json    # Execution results
    │
    └── profiler_results/                  # Neuron Profiler output
        └── {paper_id}_{timestamp}/
            ├── ntrace.pb                  # Main trace file
            ├── trace_info.pb              # Trace metadata
            └── ...
```

## Monitoring & Metrics

### CloudWatch Metrics

Training metrics are automatically logged to CloudWatch (namespace: `Trainium/Training`).

**Viewing Metrics in CloudWatch Console:**
1. AWS Console → CloudWatch → Metrics → All metrics
2. Search for: `Trainium/Training`
3. Select a metric (e.g., `execution_success`)
4. Filter by `PaperId` dimension
5. View execution status:
   - `execution_success = 1.0` → ✅ Execution completed successfully
   - `execution_success = 0.0` → ❌ Execution failed
   - No datapoints → ⏳ Execution still running or not started

**Metrics Available:**
- `training_loss` - Training loss per epoch/step
- `test_accuracy` - Test accuracy
- `execution_time_seconds` - Total execution time
- `execution_success` - 1.0 if successful, 0.0 if failed
- `peak_memory_mb` - Peak memory usage
- `estimated_cost_usd` - Estimated cost per execution

### Neuron Profiler

Hardware-level profiling is automatically enabled when `NEURON_PROFILER_ENABLED=true`.

**Profiler Output:**
- Location: `/tmp/neuron_profiler/{paper_id}_{timestamp}/` on Trainium
- Files: `ntrace.pb` (main trace), `trace_info.pb` (metadata)

**Viewing Profiler Results:**

1. **Perfetto UI (Recommended)**:
   ```bash
   # Convert to Perfetto format on Trainium
   cd /tmp/neuron_profiler/{paper_id}_{timestamp}/i-*/pid_*/
   neuron-profile view --session-dir=. --output-format=perfetto --output-file=profile.pftrace
   
   # Download and view
   scp -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129:/tmp/neuron_profiler/.../profile.pftrace ./
   # Then open https://ui.perfetto.dev/ and upload the .pftrace file
   ```

2. **TensorBoard** (requires tensorboard-plugin-neuronx):
   ```bash
   # On Trainium
   tensorboard --logdir /tmp/neuron_profiler/{paper_id}_{timestamp} --port 6006
   
   # From local (with SSH tunnel)
   ssh -i ~/.ssh/trainium-deploy-key.pem -L 6006:localhost:6006 -N ec2-user@3.21.7.129
   # Then open http://localhost:6006
   ```

3. **Check Results Locally**:
   ```bash
   python view_profiler_results.py <paper_id>
   ```

### Trainium Executor Logs

**View logs on Trainium:**
```bash
ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129
tail -f ~/trainium-executor/logs/trainium-executor.log
```

**What to look for:**
- `Executing code for paper {paper_id}` → Execution started
- `Paper {paper_id} executed successfully` → Execution completed
- `Neuron Profiler enabled` → Profiler is running
- `Logged metrics to CloudWatch` → Metrics were sent

**Quick Status Check:**
```bash
# Check if execution is running/completed
grep "{paper_id}" ~/trainium-executor/logs/trainium-executor.log | tail -20

# Check profiler output
ls -la /tmp/neuron_profiler/{paper_id}_*
```

## Troubleshooting

### Execution Timeouts

- Increase `TRAINIUM_EXECUTION_TIMEOUT` (default: 3600 seconds)
- Neuron compilation can take 20-40+ minutes for large models
- Check Trainium logs to see if execution is still running

### No Profiler Results

1. Check if profiler is enabled: `echo $NEURON_PROFILER_ENABLED` (should be `true`)
2. Check profiler output directory: `ls -la /tmp/neuron_profiler/`
3. Check executor logs for profiler messages

### No CloudWatch Metrics

1. Check if metrics logging is enabled: `echo $SAGEMAKER_METRICS_ENABLED`
2. Verify IAM permissions: Trainium instance needs `cloudwatch:PutMetricData`
3. Metrics appear after execution completes (may take 1-2 minutes)

### Connection Issues

1. Verify Trainium executor is running: `curl http://localhost:8000/health` (on Trainium)
2. Check `TRAINIUM_ENDPOINT` is correct
3. Verify security groups allow access

## Key Features

- ✅ **Neuron SDK Integration**: Generated code uses `torch_xla` for Trainium compatibility
- ✅ **Hardware-Level Profiling**: Neuron Profiler captures hardware execution traces
- ✅ **CloudWatch Metrics**: Automatic logging of training and execution metrics
- ✅ **EC2 Auto-Management**: Automatically starts/stops Trainium instance
- ✅ **Code Review**: Automated review ensures Neuron SDK compatibility



### View Profiler Output
1. `ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129`
2. `tensorboard --logdir /tmp/neuron_profiler/<PAPER ID> --port 6006`
3. `ssh -i ~/.ssh/trainium-deploy-key.pem -L 6006:localhost:6006 -N ec2-user@3.21.7.129`
4. open http://localhost:6006 in browser

