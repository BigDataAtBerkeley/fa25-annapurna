#!/usr/bin/env python3
"""
Label PDF pages using Claude 3.5 Sonnet via Bedrock.
Processes pages in batches for efficiency while ensuring independence.
"""

import os
import json
import csv
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
BATCH_SIZE = 10  # Number of pages to process per API call
MAX_RETRIES = 5
BASE_DELAY = 1  # Initial delay in seconds
MAX_DELAY = 60  # Maximum delay in seconds


class PDFPageLabeler:
    """Label PDF pages using Claude 3.5 via Bedrock."""
    
    def __init__(self, region: str = "us-east-1"):
        """Initialize Bedrock client."""
        self.region = region
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = MODEL_ID
        logger.info(f"Initialized PDF page labeler with model: {self.model_id}")
    
    def label_pages_batch(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Label a batch of pages using Claude 3.5.
        
        Args:
            pages: List of page dictionaries with 'text' field and other metadata
            
        Returns:
            List of page dictionaries with 'keep' field added (0 or 1)
        """
        if not pages:
            return []
        
        # Build prompt with clear separation between pages
        prompt = """You are labeling pages of machine learning research papers.

Your task:
Decide whether EACH page should be INCLUDED as context when generating a PyTorch implementation of the paper's method.

CRITICAL: Process each page INDEPENDENTLY. Each page's label should be based ONLY on that page's content, not on other pages in this batch.

Label as 1 (INCLUDE) ONLY if the page contains implementation-relevant content, such as:
- model architecture
- algorithmic steps or pseudocode
- training or inference procedures
- loss functions or optimization details

Label as 0 (EXCLUDE) if the page contains ONLY:
- proofs or theoretical analysis
- experimental results or tables
- background, related work, or citations
- discussion without implementation details

You will receive multiple pages. For EACH page, respond with a JSON object in this format:
{"page_index": <0-based index>, "label": 0 or 1}

Return a JSON array with one object per page, in order:
[
  {"page_index": 0, "label": 0 or 1},
  {"page_index": 1, "label": 0 or 1},
  ...
]

Pages to label:
"""
        
        # Add each page with clear separators
        for idx, page in enumerate(pages):
            page_text = page.get('text', '').strip()
            if not page_text:
                page_text = "[Empty page]"
            
            prompt += f"\n{'='*80}\n"
            prompt += f"PAGE {idx} (page_index={idx}):\n"
            prompt += f"{'='*80}\n"
            prompt += f"{page_text}\n"
        
        prompt += f"\n{'='*80}\n"
        prompt += "Remember: Return a JSON array with one object per page, using page_index to match each label to its page."
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.1,  # Low temperature for consistent labeling
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # Retry logic with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json"
                )
                
                response_body = json.loads(response["body"].read())
                generated_text = response_body["content"][0]["text"]
                
                # Parse JSON response
                labels = self._parse_response(generated_text, len(pages))
                
                # Add labels to pages
                for idx, page in enumerate(pages):
                    page['keep'] = labels.get(idx, 0)  # Default to 0 if parsing fails
                
                return pages
                
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                error_msg = str(e)
                
                # Determine if error is retryable
                retryable_errors = [
                    "ThrottlingException",
                    "ServiceUnavailableException",
                    "TooManyRequestsException",
                    "InternalServerError",
                    "ModelTimeoutException"
                ]
                is_retryable = (
                    error_code in retryable_errors or
                    "throttl" in error_msg.lower() or
                    "too many requests" in error_msg.lower() or
                    "rate limit" in error_msg.lower() or
                    "rate exceeded" in error_msg.lower() or
                    "service unavailable" in error_msg.lower()
                )
                
                if is_retryable and attempt < MAX_RETRIES - 1:
                    # Exponential backoff with jitter: delay = base * 2^attempt + random(0, 1)
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    jitter = time.time() % 1  # Add jitter to prevent thundering herd
                    total_delay = delay + jitter
                    
                    logger.warning(
                        f"Retryable error labeling batch (attempt {attempt + 1}/{MAX_RETRIES}): {error_code}. "
                        f"Retrying in {total_delay:.2f}s (exponential backoff)..."
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    if not is_retryable:
                        logger.error(f"Non-retryable error: {error_code} - {error_msg}")
                    else:
                        logger.error(f"Failed to label batch after {MAX_RETRIES} attempts: {error_code} - {error_msg}")
                    # Return pages with default label 0 on failure
                    for page in pages:
                        page['keep'] = 0
                    return pages
                    
            except Exception as e:
                logger.error(f"Unexpected error labeling batch (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff for unexpected errors too
                    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                    jitter = time.time() % 1
                    total_delay = delay + jitter
                    logger.warning(f"Retrying in {total_delay:.2f}s...")
                    time.sleep(total_delay)
                    continue
                else:
                    # Return pages with default label 0 on failure
                    for page in pages:
                        page['keep'] = 0
                    return pages
    
    def _parse_response(self, response_text: str, expected_count: int) -> Dict[int, int]:
        """
        Parse Claude's response to extract labels.
        
        Args:
            response_text: Raw response from Claude
            expected_count: Expected number of pages
            
        Returns:
            Dictionary mapping page_index to label (0 or 1)
        """
        labels = {}
        
        try:
            # Try to extract JSON array from response
            # Look for JSON array pattern
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            page_idx = item.get('page_index')
                            label = item.get('label')
                            if page_idx is not None and label is not None:
                                labels[int(page_idx)] = int(label)
            
            # If we didn't get all labels, try to find individual JSON objects
            if len(labels) < expected_count:
                json_objects = re.findall(r'\{[^}]*"page_index"[^}]*"label"[^}]*\}', response_text)
                for obj_str in json_objects:
                    try:
                        obj = json.loads(obj_str)
                        page_idx = obj.get('page_index')
                        label = obj.get('label')
                        if page_idx is not None and label is not None:
                            labels[int(page_idx)] = int(label)
                    except:
                        pass
            
            # Fallback: look for simple label patterns if JSON parsing fails
            if len(labels) < expected_count:
                logger.warning(f"Only parsed {len(labels)}/{expected_count} labels from JSON. Trying fallback parsing...")
                # Look for patterns like "page_index: 0, label: 1" or "page 0: 1"
                for idx in range(expected_count):
                    if idx not in labels:
                        # Try to find label for this index
                        pattern = rf'page[_\s]*index[_\s]*{idx}[^0-9]*label[^0-9]*([01])'
                        match = re.search(pattern, response_text, re.IGNORECASE)
                        if match:
                            labels[idx] = int(match.group(1))
                        else:
                            # Last resort: look for any mention of this index with 0 or 1
                            pattern = rf'[{idx}][^0-9]*([01])'
                            matches = re.findall(pattern, response_text)
                            if matches:
                                labels[idx] = int(matches[-1])  # Use last match
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
        
        # Ensure we have labels for all pages (default to 0)
        for idx in range(expected_count):
            if idx not in labels:
                logger.warning(f"No label found for page_index {idx}, defaulting to 0")
                labels[idx] = 0
        
        return labels


def _save_progress(output_csv: str, all_rows: List[Dict[str, Any]], fieldnames: List[str], processed_indices: set):
    """
    Save progress to CSV, writing only processed rows.
    
    Args:
        output_csv: Path to output CSV file
        all_rows: All rows (including unprocessed ones)
        fieldnames: CSV column names
        processed_indices: Set of indices that have been processed
    """
    # Create a copy with only processed rows having 'keep' field
    rows_to_write = []
    for idx, row in enumerate(all_rows):
        row_copy = row.copy()
        # Only include 'keep' if this row has been processed
        if idx in processed_indices:
            if 'keep' not in row_copy:
                row_copy['keep'] = 0
        else:
            # Remove 'keep' if not processed yet
            row_copy.pop('keep', None)
        rows_to_write.append(row_copy)
    
    # Write to temporary file first, then rename (atomic write)
    temp_file = output_csv + '.tmp'
    with open(temp_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)
    
    # Atomic rename
    import shutil
    shutil.move(temp_file, output_csv)


def process_csv(
    input_csv: str,
    output_csv: str,
    batch_size: int = BATCH_SIZE,
    max_workers: int = 3,
    start_row: int = 0,
    max_rows: Optional[int] = None
):
    """
    Process CSV file and label pages using Claude 3.5.
    Saves progress after each batch to allow resuming from interruptions.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        batch_size: Number of pages to process per API call
        max_workers: Number of concurrent batch requests
        start_row: Row to start from (for resuming)
        max_rows: Maximum number of rows to process (None = all)
    """
    logger.info(f"Loading CSV: {input_csv}")
    
    # Read ALL rows from input CSV
    all_rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        all_rows = list(reader)
    
    logger.info(f"Loaded {len(all_rows)} total rows from input CSV")
    
    # Check if output CSV exists and load progress
    processed_indices = set()
    if os.path.exists(output_csv):
        logger.info(f"Found existing output CSV: {output_csv}")
        logger.info("Loading progress from existing file...")
        try:
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = list(reader.fieldnames)
                for idx, row in enumerate(reader):
                    if 'keep' in row and row['keep'].strip():
                        try:
                            keep_val = int(row['keep'])
                            if keep_val in [0, 1]:
                                processed_indices.add(idx)
                                # Update all_rows with existing label
                                if idx < len(all_rows):
                                    all_rows[idx]['keep'] = keep_val
                        except ValueError:
                            pass
            logger.info(f"Found {len(processed_indices)} already processed rows")
        except Exception as e:
            logger.warning(f"Could not load progress from existing file: {e}")
            logger.info("Starting fresh...")
    
    # Determine which rows to process
    rows_to_process = []
    indices_to_process = []
    for idx in range(start_row, len(all_rows)):
        if idx in processed_indices:
            continue  # Skip already processed rows
        if max_rows and len(rows_to_process) >= max_rows:
            break
        rows_to_process.append(all_rows[idx])
        indices_to_process.append(idx)
    
    logger.info(f"Rows to process: {len(rows_to_process)} (starting from row {start_row})")
    
    if not rows_to_process:
        logger.info("No rows to process - all done!")
        # Still write final CSV
        if 'keep' not in fieldnames:
            fieldnames = list(fieldnames) + ['keep']
        _save_progress(output_csv, all_rows, fieldnames, processed_indices)
        return
    
    # Ensure 'keep' column exists in fieldnames
    if 'keep' not in fieldnames:
        fieldnames = list(fieldnames) + ['keep']
    
    # Initialize labeler
    labeler = PDFPageLabeler()
    
    # Process in batches
    total_batches = (len(rows_to_process) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(rows_to_process)} pages in {total_batches} batches (batch_size={batch_size})")
    
    processed_count = 0
    start_time = time.time()
    
    try:
        # Sequential processing (easier to save progress after each batch)
        for batch_idx, batch_start in enumerate(range(0, len(rows_to_process), batch_size)):
            batch_end = min(batch_start + batch_size, len(rows_to_process))
            batch = rows_to_process[batch_start:batch_end]
            batch_indices = indices_to_process[batch_start:batch_end]
            
            # Label batch
            labeled_batch = labeler.label_pages_batch(batch)
            
            # Update all_rows with labels
            for i, labeled_page in enumerate(labeled_batch):
                idx = batch_indices[i]
                all_rows[idx] = labeled_page
                processed_indices.add(idx)
            
            processed_count += len(labeled_batch)
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            
            # Save progress after each batch
            _save_progress(output_csv, all_rows, fieldnames, processed_indices)
            
            logger.info(
                f"✓ Processed batch {batch_idx + 1}/{total_batches} "
                f"({processed_count}/{len(rows_to_process)} pages, {rate:.1f} pages/sec) - Saved to {output_csv}"
            )
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user. Saving progress...")
        _save_progress(output_csv, all_rows, fieldnames, processed_indices)
        logger.info(f"Progress saved to {output_csv}")
        logger.info(f"You can resume by running with --start-row {max(indices_to_process) + 1 if indices_to_process else start_row}")
        raise
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.info("Saving progress before exiting...")
        _save_progress(output_csv, all_rows, fieldnames, processed_indices)
        raise
    
    # Print summary
    keep_count = sum(1 for idx in processed_indices if int(all_rows[idx].get('keep', 0)) == 1)
    exclude_count = len(processed_indices) - keep_count
    
    logger.info("="*80)
    logger.info("Labeling complete!")
    logger.info(f"Total pages processed: {len(processed_indices)}")
    logger.info(f"Labeled as INCLUDE (1): {keep_count} ({100*keep_count/len(processed_indices):.1f}%)")
    logger.info(f"Labeled as EXCLUDE (0): {exclude_count} ({100*exclude_count/len(processed_indices):.1f}%)")
    logger.info(f"Output saved to: {output_csv}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Label PDF pages using Claude 3.5 Sonnet via Bedrock"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV file with 'text' column (e.g., pdf_texts.csv)"
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV file with 'keep' column added"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of pages to process per API call (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent batch requests (default: 1 for sequential processing with live updates)"
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Row to start from (for resuming interrupted runs, default: 0)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (default: all)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV not found: {args.input_csv}")
        return 1
    
    # Process CSV
    try:
        process_csv(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            start_row=args.start_row,
            max_rows=args.max_rows
        )
        return 0
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Partial results may be saved.")
        return 1
    except Exception as e:
        logger.error(f"Error processing CSV: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

