#!/usr/bin/env python3
"""
View profiler results for a paper
"""

import json
import os
import sys
import glob
from pathlib import Path

paper_id = sys.argv[1] if len(sys.argv) > 1 else "_uhTW5oBclM7MZc3cZLX"

print("=" * 80)
print(f"Profiler Results for Paper: {paper_id}")
print("=" * 80)
print()

# Check local profiler results
local_profiler_dir = f"profiler_results/{paper_id}"
if os.path.exists(local_profiler_dir):
    print("📁 Local Profiler Files (downloaded from Trainium):")
    for root, dirs, files in os.walk(local_profiler_dir):
        level = root.replace(local_profiler_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            size_mb = size / (1024 * 1024)
            print(f"{subindent}{file} ({size_mb:.2f} MB)")
    print()

# Check execution result for profiler info
exec_files = sorted(glob.glob(f"results-for-deliv/trn-execution/{paper_id}_*.json"), reverse=True)
if exec_files:
    with open(exec_files[0]) as f:
        exec_result = json.load(f)
    
    profiler_info = exec_result.get("execution_result", {}).get("profiler")
    
    if profiler_info:
        print("✅ Profiler Results in Execution Output:")
        print(f"   Output Directory: {profiler_info.get('profiler_output_dir', 'N/A')}")
        print(f"   Total Files: {profiler_info.get('total_files', 0)}")
        print(f"   Files:")
        for file in profiler_info.get('profiler_files', [])[:10]:  # Show first 10
            print(f"      - {file}")
        if len(profiler_info.get('profiler_files', [])) > 10:
            print(f"      ... and {len(profiler_info.get('profiler_files', [])) - 10} more files")
        print()
    else:
        print("⚠️  Profiler field is null in execution results")
        print("   (This means the executor didn't find profiler files)")
        print()

# Check Trainium location
print("🔍 To check profiler results on Trainium:")
print(f"   ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129")
print(f"   ls -lhR /tmp/neuron_profiler/{paper_id}_*")
print()

# Profiler file types
print("📊 Profiler File Types:")
print("   - ntrace.pb: Main trace file with hardware execution traces (largest file)")
print("   - trace_info.pb: Metadata about the trace")
print("   - cpu_util.pb: CPU utilization data")
print("   - host_mem.pb: Host memory usage data")
print()
print("💡 Note: To view detailed profiler data, you may need to:")
print("   1. Use neuron-profile view (requires compatible version)")
print("   2. Or analyze the .pb files programmatically")
print("   3. Or check CloudWatch metrics for high-level performance data")
print()

print("=" * 80)

