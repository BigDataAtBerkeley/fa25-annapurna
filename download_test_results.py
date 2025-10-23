#!/usr/bin/env python3
"""
Download test results from Trainium execution stored in S3.
Test results include stdout, stderr, plots, and execution metrics.
"""

import boto3
import sys
import json
from pathlib import Path
from datetime import datetime

s3 = boto3.client('s3')
BUCKET = 'papers-test-outputs'
OUTPUT_DIR = 'test_results'

def list_tested_papers():
    """List all papers with test results."""
    response = s3.list_objects_v2(Bucket=BUCKET, Delimiter='/')
    
    if 'CommonPrefixes' not in response:
        print("No test results found in S3")
        return []
    
    papers = []
    print("\nPapers with test results:")
    print("=" * 80)
    
    for idx, prefix in enumerate(response['CommonPrefixes'], 1):
        paper_id = prefix['Prefix'].rstrip('/')
        
        try:
            # execution metadata
            outputs_prefix = f"{paper_id}/outputs/"
            objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=outputs_prefix)
            
            has_stdout = False
            has_stderr = False
            has_plots = False
            num_plots = 0
            
            if 'Contents' in objects:
                for obj in objects['Contents']:
                    key = obj['Key']
                    if 'stdout.txt' in key:
                        has_stdout = True
                    if 'stderr.txt' in key:
                        has_stderr = True
                    if 'plots/' in key and key.endswith(('.png', '.jpg', '.pdf')):
                        has_plots = True
                        num_plots += 1
            
            print(f"{idx}. Paper ID: {paper_id}")
            print(f"   Outputs: stdout={has_stdout}, stderr={has_stderr}, plots={num_plots}")
            print()
            
        except Exception as e:
            print(f"{idx}. Paper ID: {paper_id} (error reading metadata: {e})")
            print()
        
        papers.append(paper_id)
    
    return papers


def download_test_results(paper_id: str):
    """Download all test results for a specific paper."""
    output_path = Path(OUTPUT_DIR) / paper_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading test results for {paper_id}...")
    
    downloaded = []
    
    try:
        # List all objects for this paper
        outputs_prefix = f"{paper_id}/outputs/"
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=outputs_prefix)
        
        if 'Contents' not in response:
            print(f"No test results found for {paper_id}")
            return False
        
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directory markers
            if key.endswith('/'):
                continue
            
            # Get filename from key
            filename = key.replace(outputs_prefix, '')
            
            # Create subdirectories if needed (e.g., for plots/)
            local_path = output_path / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            try:
                s3.download_file(BUCKET, key, str(local_path))
                size_kb = obj['Size'] / 1024
                print(f"  ✅ {filename} ({size_kb:.1f} KB)")
                downloaded.append(filename)
            except Exception as e:
                print(f"  ❌ {filename} - Error: {e}")
        
        print(f"\n✅ Downloaded {len(downloaded)} files to {output_path}/")
        
        # Display summary if stdout exists
        stdout_path = output_path / "stdout.txt"
        stderr_path = output_path / "stderr.txt"
        
        if stdout_path.exists():
            print(f"\n{'='*60}")
            print("📄 STDOUT Preview (last 20 lines):")
            print(f"{'='*60}")
            with open(stdout_path, 'r') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(line.rstrip())
        
        if stderr_path.exists() and stderr_path.stat().st_size > 0:
            print(f"\n{'='*60}")
            print("⚠️  STDERR Preview (last 20 lines):")
            print(f"{'='*60}")
            with open(stderr_path, 'r') as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(line.rstrip())
        
        return True
        
    except Exception as e:
        print(f"Error downloading test results: {e}")
        return False


def download_all_results(papers):
    """Download test results for all papers."""
    print(f"\n📥 Downloading results for {len(papers)} papers...\n")
    
    success_count = 0
    for i, paper_id in enumerate(papers, 1):
        print(f"\n[{i}/{len(papers)}] {paper_id}")
        if download_test_results(paper_id):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Successfully downloaded {success_count}/{len(papers)} test results")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"{'='*80}")


def show_summary():
    """Show summary statistics of test results."""
    try:
        response = s3.list_objects_v2(Bucket=BUCKET, Delimiter='/')
        
        if 'CommonPrefixes' not in response:
            print("No test results found")
            return
        
        total_papers = len(response['CommonPrefixes'])
        successful = 0
        failed = 0
        
        for prefix in response['CommonPrefixes']:
            paper_id = prefix['Prefix'].rstrip('/')
            stdout_key = f"{paper_id}/outputs/stdout.txt"
            
            try:
                s3.head_object(Bucket=BUCKET, Key=stdout_key)
                successful += 1
            except:
                failed += 1
        
        print(f"\n{'='*60}")
        print("Test Results Summary")
        print(f"{'='*60}")
        print(f"Total tested papers: {total_papers}")
        print(f"Successful tests:    {successful}")
        print(f"Failed tests:        {failed}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error getting summary: {e}")


def main():
    print("="*80)
    print("Download Trainium Test Results from S3")
    print("="*80)
    
    show_summary()
    papers = list_tested_papers()
    
    if not papers:
        return
    
    # Allow user to choose which results to download
    try:
        print("\nOptions:")
        print("  • Enter number (1-{}) to download specific paper".format(len(papers)))
        print("  • Enter 'all' to download all results")
        print("  • Enter 'q' to quit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice.lower() == 'q':
            print("Cancelled")
            return
        elif choice.lower() == 'all':
            download_all_results(papers)
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(papers):
                paper_id = papers[idx]
                download_test_results(paper_id)
            else:
                print("Invalid choice")
                
    except KeyboardInterrupt:
        print("\n\nCancelled")
    except ValueError:
        print("Invalid input")


if __name__ == "__main__":
    main()

