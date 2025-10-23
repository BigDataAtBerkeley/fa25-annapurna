# Trainium Executor Service

HTTP service that runs on AWS Trainium instances (trn1.2xlarge) to execute PyTorch code with hardware acceleration.

## Overview
This service receives batches of PyTorch code from the `test_lambda` Lambda function, executes them using AWS Neuron SDK on Trainium hardware, and returns detailed execution results.

## Architecture

```
Lambda (test_lambda)  →  HTTP POST  →  Trainium Service (this)
                                            ↓
                                      Execute PyTorch Code
                                            ↓
                                       Return Results
```

## Files

- **`app.py`** - Flask-based HTTP server for code execution
- **`requirements.txt`** - Python dependencies


## Setup on Trainium Instance

### Prerequisites
- AWS trn1 instance
- Deep Learning AMI 
- SSH access to the instance

### Installation

1. SSH into your Trainium instance (GET PUBLIC_IP FROM DAN):
```bash
ssh -i your-key.pem ubuntu@<TRAINIUM_PUBLIC_IP> 
```

2. Deploy using the automated script:
```bash
# From the annapurna directory
cd deployment
./deploy_trainium.sh /path/to/your-key.pem
```

Or manually copy files and run setup:
```bash
# Files are uploaded to S3, download them on the instance:
aws s3 cp s3://papers-test-outputs/setup/app.py ~/trainium-executor/
aws s3 cp s3://papers-test-outputs/setup/requirements.txt ~/trainium-executor/
aws s3 cp s3://papers-test-outputs/setup/setup_trainium_remote.sh ./
chmod +x setup_trainium_remote.sh
./setup_trainium_remote.sh
```

This will:
- Install Neuron SDK and drivers
- Set up Python environment with PyTorch-Neuron
- Create systemd service
- Start the service on port 8000

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-23T10:30:00",
  "neuron_available": true,
  "working_dir": "/var/lib/trainium-jobs",
  "max_execution_time": 600
}
```

### `POST /execute_batch`
Execute a batch of code files.

**Request:**
```json
{
  "batch": [
    {
      "paper_id": "paper_123",
      "paper_title": "ResNet Paper",
      "code": "import torch\n# PyTorch code here...",
      "s3_code_key": "paper_123/code.py"
    }
  ],
  "timeout": 600
}
```

**Response:**
```json
{
  "success": true,
  "batch_size": 10,
  "successful": 8,
  "failed": 2,
  "results": {
    "paper_123": {
      "success": true,
      "execution_time": 45.2,
      "return_code": 0,
      "stdout": "Training complete...",
      "stderr": "",
      "timeout": false,
      "lines_of_code": 342,
      "has_training_loop": true,
      "peak_memory_mb": 12500.5,
      "training_loss": 0.023,
      "validation_accuracy": 0.945
    }
  }
}
```

### `POST /execute`
Execute a single code file (for testing).

**Request:**
```json
{
  "paper_id": "paper_123",
  "code": "import torch\nprint('Hello Trainium')",
  "timeout": 600
}
```

## Metrics Extraction

To return metrics from your generated code, print a line starting with `METRICS:` followed by JSON:

```python
import json

# Your training code here...
final_loss = 0.023
accuracy = 0.945

# Print metrics for collection
print(f"METRICS: {json.dumps({'training_loss': final_loss, 'validation_accuracy': accuracy})}")
```

The service will parse this and include it in the response.

## Service Management

```bash
# View logs
sudo journalctl -u trainium-executor -f
# Restart service
sudo systemctl restart trainium-executor
# Stop service
sudo systemctl stop trainium-executor
# Check status
sudo systemctl status trainium-executor
# Test health
curl http://localhost:8000/health
```

## Network Config

### Lambda Configuration
`test_lambda` Lambda function has to:
- Be deployed in the **same VPC** as the Trainium instance (Alrdy configured this way)
- Have `TRAINIUM_ENDPOINT` environment variable set to: `http://<PRIVATE_IP>:8000`


### Neuron runtime issues
```bash
# Check Neuron status
neuron-ls
# Check Neuron tools
neuron-top
```

### Execution timeouts
Adjust the timeout in `test_lambda` environment variables or in the request payload.

## Cost Optimization

- **Auto-stop when idle**: Set up CloudWatch alarm to stop instance after idle period
- **Use Spot instances**: For non-critical workloads
- **Instance scheduling**: Use EventBridge to start/stop on schedule

## Performance Tuning

- **Batch size**: Larger batches (up to 10) reduce per-paper overhead
- **Execution timeout**: Adjust based on typical code complexity
- **Neuron cores**: trn1.2xlarge has 2 Neuron cores; optimize code for multi-core usage

## Testing Locally

```bash
# Start service
python3 app.py

# Test health endpoint
curl http://localhost:8000/health

# Test execution
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "paper_id": "test_123",
    "code": "import torch\nprint(\"Hello from Trainium\")\nprint(torch.__version__)"
  }'
```

