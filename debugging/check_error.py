#!/usr/bin/env python3
"""
Quick script to check execution errors for a paper.
Can be run locally (checks Trainium endpoint) or on Trainium (checks DynamoDB).
"""

import sys
import json
import requests
import os

def check_via_status_endpoint(paper_id: str, endpoint: str = "http://3.21.7.129:8000"):
    """Check error via Trainium status endpoint."""
    try:
        url = f"{endpoint}/status/{paper_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Error Count: {data.get('error_count', 0)}")
        
        latest_error = data.get('latest_error', {})
        if latest_error:
            print(f"\n=== Latest Error ===")
            print(f"Type: {latest_error.get('error_type', 'N/A')}")
            print(f"Return Code: {latest_error.get('return_code', 'N/A')}")
            print(f"Execution Time: {latest_error.get('execution_time', 'N/A')}s")
            
            stderr = latest_error.get('stderr', '')
            if stderr:
                print(f"\n=== Full STDERR ===")
                print(stderr)
                
                # Extract actual errors (skip warnings)
                print(f"\n=== Actual Errors (filtered) ===")
                lines = stderr.split('\n')
                in_traceback = False
                for line in lines:
                    # Skip warnings
                    if any(w in line.lower() for w in ['warning', 'deprecation', 'pjrt', 'c-api']):
                        continue
                    # Show traceback and errors
                    if 'Traceback' in line or 'Error:' in line or 'Exception:' in line:
                        in_traceback = True
                        print(line)
                    elif in_traceback:
                        if line.strip() and not line.startswith(' '):
                            in_traceback = False
                        if in_traceback or any(err in line for err in ['Error', 'Exception', 'Failed']):
                            print(line)
            
            stdout = latest_error.get('stdout', '')
            if stdout:
                print(f"\n=== STDOUT (last 500 chars) ===")
                print(stdout[-500:])
    except Exception as e:
        print(f"Error checking status endpoint: {e}")

def check_via_dynamodb(paper_id: str):
    """Check errors via DynamoDB"""
    try:
        # Add trn_execute to path so we can import error_db
        import sys
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        trn_execute_dir = os.path.join(os.path.dirname(script_dir), 'trn_execute')
        if trn_execute_dir not in sys.path:
            sys.path.insert(0, trn_execute_dir)
        
        from error_db import get_errors
        
        errors = get_errors(paper_id)
        print(f"Found {len(errors)} errors for {paper_id}\n")
        
        for i, err in enumerate(errors, 1):
            err_data = err.get('error_data', {})
            iteration = err.get('iteration', 'N/A')
            timestamp = err.get('timestamp', 'N/A')
            print(f"{'='*60}")
            print(f"Error {i}/{len(errors)}")
            print(f"{'='*60}")
            print(f"Iteration: {iteration}")
            print(f"Timestamp: {timestamp}")
            print(f"Type: {err_data.get('error_type', 'N/A')}")
            print(f"Return Code: {err_data.get('return_code', 'N/A')}")
            print(f"Execution Time: {err_data.get('execution_time', 'N/A')}s")
            
            # Show fixes_applied if available
            fixes_applied = err.get('fixes_applied')
            if fixes_applied:
                print(f"\nFixes Applied:")
                if isinstance(fixes_applied, dict):
                    fixes_list = fixes_applied.get('fixes', [])
                    if fixes_list:
                        for fix in fixes_list[:5]:  # Show first 5 fixes
                            print(f"  - {fix}")
                elif isinstance(fixes_applied, list):
                    for fix in fixes_applied[:5]:
                        print(f"  - {fix}")
                else:
                    print(f"  {fixes_applied}")
            print()
            
            stderr = err_data.get('stderr', '')
            if stderr:
                print("=== Full STDERR ===")
                print(stderr)
                print()
                
                # Extract actual errors
                print("=== Actual Errors (filtered) ===")
                lines = stderr.split('\n')
                in_traceback = False
                for line in lines:
                    # Skip warnings
                    if any(w in line.lower() for w in ['warning', 'deprecation', 'pjrt', 'c-api', 'level=info']):
                        continue
                    # Show traceback and errors
                    if 'Traceback' in line or any(err in line for err in ['Error:', 'Exception:', 'RuntimeError', 'ValueError', 'TypeError', 'AttributeError']):
                        in_traceback = True
                        print(line)
                    elif in_traceback:
                        if line.strip() and not line.startswith(' ') and 'Error' not in line:
                            in_traceback = False
                        if in_traceback or any(err in line for err in ['Error', 'Exception', 'Failed', 'raise']):
                            print(line)
                print()
            
            stdout = err_data.get('stdout', '')
            if stdout:
                print("=== STDOUT (last 500 chars) ===")
                print(stdout[-500:])
                print()
            
            print()
    except ImportError:
        print("Error: Cannot import error_db. This script must be run on Trainium.")
        print("   Or use --endpoint to check via HTTP endpoint.")
    except Exception as e:
        print(f"Error checking DynamoDB: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_error.py <paper_id> [--endpoint <url>]")
        print("  --endpoint: Check via HTTP endpoint (default: http://3.21.7.129:8000)")
        print("  Without --endpoint: Check via DynamoDB (must be run on Trainium)")
        sys.exit(1)
    
    paper_id = sys.argv[1]
    
    # Check if endpoint is specified
    if '--endpoint' in sys.argv:
        idx = sys.argv.index('--endpoint')
        endpoint = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "http://3.21.7.129:8000"
        check_via_status_endpoint(paper_id, endpoint)
    else:
        # Try DynamoDB first, fallback to endpoint
        try:
            check_via_dynamodb(paper_id)
        except:
            print("Falling back to status endpoint...")
            check_via_status_endpoint(paper_id)

