#!/usr/bin/env python3
"""
Check status of all papers that were submitted to Trainium executor.
"""

import os
import sys
import requests
import json
from typing import Dict, List, Any

TRAINIUM_ENDPOINT = os.getenv('TRAINIUM_ENDPOINT', 'http://3.21.7.129:8000')

# Paper IDs from the log
PAPER_IDS = [
    'Feg3dpoBclM7MZc3EpWP',
    'V-g6dpoBclM7MZc3x5eR',
    'cOg6dpoBclM7MZc34pfp',
    '-eg4dpoBclM7MZc3y5WS',
    'eeg5dpoBclM7MZc3oJZO',
    'reg2dpoBclM7MZc3cJTJ',
    '4OhJW5oBclM7MZc3K5Iq',
    '6eg2dpoBclM7MZc3t5Ql',
    '_ug2dpoBclM7MZc3-ZSN',
    '_-hIZpoBclM7MZc3UpNE'
]

def check_paper_status(paper_id: str) -> Dict[str, Any]:
    """Check status of a single paper."""
    try:
        url = f"{TRAINIUM_ENDPOINT}/status/{paper_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e),
            "status": "error",
            "paper_id": paper_id
        }

def main():
    print(f"Checking status of {len(PAPER_IDS)} papers on Trainium executor...")
    print(f"Endpoint: {TRAINIUM_ENDPOINT}\n")
    print("=" * 80)
    
    results = []
    for paper_id in PAPER_IDS:
        status = check_paper_status(paper_id)
        results.append((paper_id, status))
        
        # Print status
        paper_status = status.get('status', 'unknown')
        error_count = status.get('error_count', 0)
        
        if paper_status == 'running':
            print(f"✅ {paper_id}: RUNNING (errors: {error_count})")
        elif paper_status == 'success':
            exec_time = status.get('execution_time', 0)
            print(f"✅ {paper_id}: SUCCESS ({exec_time:.1f}s, errors: {error_count})")
        elif paper_status == 'failed':
            latest_error = status.get('latest_error', {})
            error_msg = latest_error.get('error_message', 'Unknown error')[:100]
            print(f"❌ {paper_id}: FAILED (errors: {error_count})")
            print(f"   Error: {error_msg}")
        elif paper_status == 'error':
            print(f"❌ {paper_id}: ERROR - {status.get('error', 'Unknown')}")
        else:
            print(f"⚠️  {paper_id}: {paper_status.upper()} (errors: {error_count})")
    
    print("\n" + "=" * 80)
    
    # Summary
    running = sum(1 for _, s in results if s.get('status') == 'running')
    success = sum(1 for _, s in results if s.get('status') == 'success')
    failed = sum(1 for _, s in results if s.get('status') == 'failed')
    error = sum(1 for _, s in results if s.get('status') == 'error')
    unknown = len(results) - running - success - failed - error
    
    print(f"\nSummary:")
    print(f"  Running: {running}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Error: {error}")
    print(f"  Unknown: {unknown}")
    
    # Show details for failed papers
    if failed > 0:
        print(f"\n❌ Failed Papers Details:")
        for paper_id, status in results:
            if status.get('status') == 'failed':
                latest_error = status.get('latest_error', {})
                error_type = latest_error.get('error_type', 'unknown')
                error_msg = latest_error.get('error_message', 'No error message')[:200]
                print(f"\n  {paper_id}:")
                print(f"    Type: {error_type}")
                print(f"    Message: {error_msg}")

if __name__ == '__main__':
    main()

