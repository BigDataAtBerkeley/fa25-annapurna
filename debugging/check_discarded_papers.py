#!/usr/bin/env python3
"""
Check Discarded Papers Bucket

This script lists all papers in the discarded papers S3 bucket and displays
their structure, keys, and reasons for rejection.

Usage:
    python check_discarded_papers.py
"""

import boto3
import json
import os
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# AWS Config
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DISCARDED_BUCKET = os.getenv("DISCARDED_BUCKET", "discarded-papers")

# Initialize S3 client
s3_client = boto3.client("s3", region_name=AWS_REGION)


def get_all_discarded_papers(bucket: str) -> List[Dict[str, Any]]:
    """List all objects in the discarded papers bucket."""
    papers = []
    paginator = s3_client.get_paginator("list_objects_v2")
    
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix="rejected/"):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith(".json"):
                    papers.append({
                        "key": key,
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"]
                    })
    except s3_client.exceptions.NoSuchBucket:
        print(f"❌ Bucket '{bucket}' does not exist")
        return []
    except Exception as e:
        print(f"❌ Error listing objects: {e}")
        return []
    
    return papers


def download_and_parse_paper(bucket: str, key: str) -> Dict[str, Any]:
    """Download and parse a single discarded paper record."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}


def get_all_keys(record: Dict[str, Any], prefix: str = "") -> List[str]:
    """Recursively get all keys from a nested dictionary."""
    keys = []
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else key
        keys.append(full_key)
        if isinstance(value, dict):
            keys.extend(get_all_keys(value, full_key))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Handle list of dicts - show first item's keys
            keys.extend([f"{full_key}[0].{k}" for k in value[0].keys()])
    return keys


def display_paper_details(record: Dict[str, Any], key: str, index: int, total: int):
    """Display detailed information about a single paper."""
    print(f"\n{'='*80}")
    print(f"Paper {index + 1} of {total}")
    print(f"S3 Key: {key}")
    print(f"{'='*80}")
    
    if "error" in record:
        print(f"❌ Error loading paper: {record['error']}")
        return
    
    # Display top-level keys
    print("\n📋 Top-level keys:")
    for k in sorted(record.keys()):
        value = record[k]
        if isinstance(value, dict):
            print(f"  • {k}: (dict with {len(value)} keys)")
        elif isinstance(value, list):
            print(f"  • {k}: (list with {len(value)} items)")
        elif isinstance(value, str) and len(value) > 100:
            print(f"  • {k}: (string, {len(value)} chars)")
        else:
            print(f"  • {k}: {value}")
    
    # Display all nested keys
    all_keys = get_all_keys(record)
    print(f"\n🔑 All keys ({len(all_keys)} total):")
    for k in sorted(all_keys):
        print(f"  • {k}")
    
    # Highlight important fields
    print("\n📝 Important fields:")
    if "reason" in record:
        reason = record["reason"]
        if isinstance(reason, str) and len(reason) > 200:
            print(f"  • reason: {reason[:200]}...")
        else:
            print(f"  • reason: {reason}")
    else:
        print("  ⚠️  reason: MISSING")
    
    if "rejected_by" in record:
        print(f"  • rejected_by: {record['rejected_by']}")
    else:
        print("  ⚠️  rejected_by: MISSING")
    
    if "decision" in record:
        print(f"  • decision: {record['decision']}")
    
    if "evaluated_at" in record:
        print(f"  • evaluated_at: {record['evaluated_at']}")
    
    if "paper" in record and isinstance(record["paper"], dict):
        paper = record["paper"]
        if "title" in paper:
            title = paper["title"]
            if len(title) > 80:
                print(f"  • paper.title: {title[:80]}...")
            else:
                print(f"  • paper.title: {title}")
    
    # Show full JSON structure (truncated if too long)
    print("\n📄 Full JSON structure:")
    json_str = json.dumps(record, indent=2, default=str)
    if len(json_str) > 2000:
        print(json_str[:2000] + "\n... (truncated)")
    else:
        print(json_str)


def display_summary(papers_data: List[Dict[str, Any]]):
    """Display summary statistics about discarded papers."""
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    total = len(papers_data)
    print(f"\nTotal discarded papers: {total}")
    
    if total == 0:
        print("No papers found in the bucket.")
        return
    
    # Count by rejection type
    rejected_by_counts = defaultdict(int)
    has_reason = 0
    missing_reason = 0
    missing_rejected_by = 0
    
    for paper_data in papers_data:
        record = paper_data.get("record", {})
        if "rejected_by" in record:
            rejected_by_counts[record["rejected_by"]] += 1
        else:
            missing_rejected_by += 1
        
        if "reason" in record:
            has_reason += 1
        else:
            missing_reason += 1
    
    print(f"\n📈 Rejection breakdown:")
    for rejected_by, count in sorted(rejected_by_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total) * 100
        print(f"  • {rejected_by}: {count} ({percentage:.1f}%)")
    
    if missing_rejected_by > 0:
        print(f"  ⚠️  Missing 'rejected_by': {missing_rejected_by}")
    
    print(f"\n✅ Papers with 'reason' field: {has_reason}")
    if missing_reason > 0:
        print(f"  ⚠️  Papers missing 'reason' field: {missing_reason}")
    
    # Date range
    dates = []
    for paper_data in papers_data:
        record = paper_data.get("record", {})
        if "evaluated_at" in record:
            try:
                dates.append(datetime.fromisoformat(record["evaluated_at"].replace("Z", "+00:00")))
            except:
                pass
    
    if dates:
        dates.sort()
        print(f"\n📅 Date range:")
        print(f"  • Oldest: {dates[0].isoformat()}")
        print(f"  • Newest: {dates[-1].isoformat()}")


def main():
    parser = argparse.ArgumentParser(description="Check discarded papers in S3 bucket")
    parser.add_argument(
        "--bucket",
        type=str,
        default=DISCARDED_BUCKET,
        help=f"S3 bucket name (default: {DISCARDED_BUCKET})"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary statistics"
    )
    parser.add_argument(
        "--rejected-by",
        type=str,
        help="Filter by rejection type (e.g., 'claude', 'rag', 'exact_duplicate')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of papers to display (default: all)"
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Checking discarded papers in bucket: {args.bucket}")
    print(f"Region: {AWS_REGION}")
    
    # Get all paper keys
    paper_keys = get_all_discarded_papers(args.bucket)
    
    if not paper_keys:
        print("No papers found in the bucket.")
        return
    
    print(f"\nFound {len(paper_keys)} papers in the bucket")
    
    # Download and parse all papers
    papers_data = []
    for i, paper_info in enumerate(paper_keys):
        if args.limit and i >= args.limit:
            break
        
        record = download_and_parse_paper(args.bucket, paper_info["key"])
        
        # Apply filter if specified
        if args.rejected_by:
            if record.get("rejected_by") != args.rejected_by:
                continue
        
        papers_data.append({
            "key": paper_info["key"],
            "size": paper_info["size"],
            "last_modified": paper_info["last_modified"],
            "record": record
        })
    
    if not papers_data:
        print(f"\nNo papers found matching the criteria.")
        return
    
    # Display summary
    display_summary(papers_data)
    
    # Display details if not summary-only
    if not args.summary:
        print(f"\n\n{'='*80}")
        print(f"DETAILED VIEW ({len(papers_data)} papers)")
        print("="*80)
        
        for i, paper_data in enumerate(papers_data):
            display_paper_details(
                paper_data["record"],
                paper_data["key"],
                i,
                len(papers_data)
            )
            
            if i < len(papers_data) - 1:
                print("\n")


if __name__ == "__main__":
    main()

