# Trainium Test Executor

Local testing utility for running generated code files on Trainium instance without saving to S3 or OpenSearch.

## Overview

This tool reads Python code files from the `generated_code/` directory, sends them to your Trainium instance for execution, and saves all results locally to `trainium_test_results/`.

## Features

- ✅ Tests code files locally on Trainium
- ✅ Automatically starts/stops Trainium instance if needed
- ✅ Saves results locally (no S3/OpenSearch)
- ✅ Supports testing individual files or all files
- ✅ Saves stdout, stderr, plots, and execution summaries

## Setup

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Required
TRAINIUM_ENDPOINT=http://your-trainium-ip:8000

# Optional (for auto start/stop)
TRAINIUM_INSTANCE_ID=i-xxxxxxxxxxxxx
TRAINIUM_REGION=us-east-2
TRAINIUM_TIMEOUT=600

# Optional (custom paths)
GENERATED_CODE_DIR=generated_code
```

### Finding Your Trainium Instance ID

```bash
aws ec2 describe-instances \
  --region us-east-2 \
  --filters "Name=instance-type,Values=trn1.2xlarge" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress]' \
  --output table
```

## Usage

### Test All Files in generated_code/

```bash
python trainium_test_executor/test_on_trainium.py
```

### Test a Specific File

```bash
python trainium_test_executor/test_on_trainium.py \
  --file Diffusion_Transformers_for_Tabular_Data_Time_Serie_20251108_160358.py
```

### Custom Code Directory

```bash
python trainium_test_executor/test_on_trainium.py \
  --code-dir ./my_code
```

### Custom Timeout

```bash
python trainium_test_executor/test_on_trainium.py \
  --timeout 900
```

### Keep Instance Running After Testing

By default, the instance stops automatically after all tests. To keep it running:

```bash
python trainium_test_executor/test_on_trainium.py \
  --no-stop
```

### Stop Instance Only

```bash
python trainium_test_executor/test_on_trainium.py \
  --stop-only
```

## Output Structure

Results are saved to `trainium_test_executor/trainium_test_results/`:

```
trainium_test_results/
├── {paper_id_1}/
│   ├── code.py              # Code that was tested
│   ├── stdout.log           # Standard output
│   ├── stderr.log           # Standard error
│   ├── summary.json         # Execution summary
│   └── plots/               # Generated plots (if any)
│       └── plot1.png
├── {paper_id_2}/
│   └── ...
```

### Summary JSON Format

```json
{
  "paper_id": "NehUW5oBclM7MZc3lpNY",
  "paper_title": "Diffusion Transformers for Tabular Data",
  "success": true,
  "execution_time": 45.2,
  "return_code": 0,
  "timeout": false,
  "error_message": null,
  "error_type": null,
  "tested_at": "2025-11-08T16:30:00",
  "detailed_metrics": {
    "training_loss": 0.023,
    "validation_loss": 0.031
  }
}
```

## Instance Management

The tool automatically handles Trainium instance lifecycle:

1. **Start**: If instance is stopped, it will start it and wait for services
2. **Health Check**: Verifies Trainium endpoint is accessible
3. **Stop**: Use `--auto-stop` to stop instance after testing, or `--stop-only` to just stop

**Note**: Instance start/stop requires `TRAINIUM_INSTANCE_ID` to be set in environment.

## Requirements

- AWS credentials configured (for EC2 instance management)
- Trainium instance running Flask app (see `trainium_executor/`)
- Network access to Trainium endpoint
- Python dependencies: `boto3`, `requests`, `python-dotenv`

## Troubleshooting

### Connection Errors

If you get connection errors:
1. Check `TRAINIUM_ENDPOINT` is correct
2. Verify Trainium Flask app is running: `curl http://your-endpoint:8000/health`
3. Check security groups allow access from your IP

### Instance Not Starting

If instance doesn't start:
1. Verify `TRAINIUM_INSTANCE_ID` is correct
2. Check AWS credentials have EC2 permissions
3. Verify instance exists in the specified region

### Timeout Issues

If executions timeout:
1. Increase timeout: `--timeout 1200`
2. Check Trainium instance has enough resources
3. Verify code doesn't have infinite loops

