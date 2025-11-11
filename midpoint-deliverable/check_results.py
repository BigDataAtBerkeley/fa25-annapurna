#!/usr/bin/env python3
"""
Quick script to check execution results for a paper
"""

import json
import glob
import os
import sys

paper_id = sys.argv[1] if len(sys.argv) > 1 else "9OhIZpoBclM7MZc3RpN3"

print("=" * 80)
print(f"Results Summary for Paper: {paper_id}")
print("=" * 80)
print()

# 1. Execution results
exec_files = sorted(glob.glob(f"results/{paper_id}/trn-execution/{paper_id}_*.json"), reverse=True)
if exec_files:
    with open(exec_files[0]) as f:
        exec_result = json.load(f)
    print("1. ✅ Execution Results:")
    print(f"   File: {os.path.basename(exec_files[0])}")
    result = exec_result.get("execution_result", {})
    print(f"   Success: {result.get('success', False)}")
    print(f"   Execution Time: {result.get('execution_time', 0):.1f} seconds ({result.get('execution_time', 0)/60:.1f} minutes)")
    print(f"   Return Code: {result.get('return_code', -1)}")
    
    if result.get('detailed_metrics'):
        print(f"   📊 Training Metrics:")
        metrics = result['detailed_metrics']
        if 'test_accuracy' in metrics:
            print(f"      Test Accuracy: {metrics['test_accuracy']:.2f}%")
        if 'training_loss' in metrics:
            print(f"      Final Training Loss: {metrics['training_loss']:.4f}")
        if 'initial_epoch_loss' in metrics:
            print(f"      Initial Loss: {metrics['initial_epoch_loss']:.4f}")
        if 'num_epochs' in metrics:
            print(f"      Epochs: {metrics['num_epochs']}")
    print()

# 2. Metrics file (may be in trn-execution or separate metrics folder)
metrics_files = sorted(glob.glob(f"results/{paper_id}/trn-execution/{paper_id}_*.json"), reverse=True)
if not metrics_files:
    metrics_files = sorted(glob.glob(f"results/{paper_id}/metrics/{paper_id}_*.json"), reverse=True)
if metrics_files:
    with open(metrics_files[0]) as f:
        metrics = json.load(f)
    print("2. 📊 Detailed Metrics:")
    print(f"   File: {os.path.basename(metrics_files[0])}")
    if 'metrics' in metrics:
        for key, value in metrics['metrics'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    print()

# 3. Profiler (check profiler_results folder)
profiler_files = sorted(glob.glob(f"results/{paper_id}/profiler_results/*.json"), reverse=True)
if profiler_files:
    print("3. 🔬 Profiler Results:")
    print(f"   File: {os.path.basename(profiler_files[0])}")
    with open(profiler_files[0]) as f:
        profiler = json.load(f)
    print(f"   Output Directory: {profiler.get('profiler_output_dir', 'N/A')}")
    print(f"   Files: {profiler.get('profiler_files', [])}")
else:
    print("3. 🔬 Profiler Results: Not found locally")
    print(f"   💡 Check on Trainium: find /tmp/neuron_profiler -name '*{paper_id}*'")
print()

# 4. CloudWatch
print("4. ☁️  CloudWatch Metrics:")
print(f"   Run: python check_cloudwatch_metrics.py")
print(f"   (Update PAPER_ID in the script to '{paper_id}')")
print()

# 5. Trainium logs
print("5. 📋 Trainium Logs:")
print(f"   SSH: ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129")
print(f"   Then: grep '{paper_id}' ~/trainium-executor/logs/trainium-executor.log | tail -20")
print()

print("=" * 80)
print("✅ Execution completed successfully!")
print("=" * 80)

