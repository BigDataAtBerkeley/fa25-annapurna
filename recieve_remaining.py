import boto3
import json
import csv
import time
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

# === Configuration ===
region = os.getenv("AWS_REGION")
receive_queue_url = os.getenv("RESULTS_QUEUE_URL")
output_file = "paper_results.csv"

if not region or not receive_queue_url:
    raise EnvironmentError("Missing env vars: AWS_REGION, RESULTS_QUEUE_URL")

# === AWS Client ===
sqs = boto3.client("sqs", region_name=region)

# === Load existing results to avoid duplicates ===
existing_ids = set()
try:
    with open(output_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ids.add(row["paper_id"])
    print(f"📋 Found {len(existing_ids)} existing results in {output_file}")
except FileNotFoundError:
    print(f"⚠️ No existing file found. Will create new {output_file}")

# === Poll results queue ===
results = []
received_ids = set()
start_time = time.time()
timeout_seconds = 900  # 5 min timeout
expected_remaining = 250 - 125  # Adjust if needed

print(f"\n🔍 Polling queue for remaining results...")
print(f"Expected: ~{expected_remaining} results\n")

consecutive_empty = 0
max_empty_polls = 3  # Stop after 3 empty polls in a row

while (time.time() - start_time) < timeout_seconds and consecutive_empty < max_empty_polls:
    resp = sqs.receive_message(
        QueueUrl=receive_queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=10,
    )

    messages = resp.get("Messages", [])
    
    if not messages:
        consecutive_empty += 1
        print(f"⏳ No messages (attempt {consecutive_empty}/{max_empty_polls})...")
        continue
    
    consecutive_empty = 0  # Reset counter when we get messages

    for msg in messages:
        try:
            body = json.loads(msg["Body"])
            payload = body.get("responsePayload") or body
            paper_id = payload.get("paper_id") or f"msg_{len(results)+1}"

            # Skip if already in existing file or already received in this run
            if paper_id not in existing_ids and paper_id not in received_ids:
                results.append(payload)
                received_ids.add(paper_id)

                count = len(received_ids)
                if count == 1:
                    print("📥 Received first new result!")
                elif count % 10 == 0:
                    print(f"📈 Received {count} new results so far...")

            # Delete processed message
            sqs.delete_message(
                QueueUrl=receive_queue_url,
                ReceiptHandle=msg["ReceiptHandle"]
            )

        except Exception as e:
            print(f"⚠️ Error parsing message: {e}")
            # Still delete the problematic message
            try:
                sqs.delete_message(
                    QueueUrl=receive_queue_url,
                    ReceiptHandle=msg["ReceiptHandle"]
                )
            except:
                pass

print(f"\n✅ Polling complete! Received {len(received_ids)} new results.")

# === Append new results to CSV ===
if results:
    file_exists = os.path.exists(output_file)
    
    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["paper_id", "title", "decision", "reason", "similarity", "similar_paper"]
        )
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        for r in results:
            writer.writerow({
                "paper_id": r.get("paper_id"),
                "title": r.get("title"),
                "decision": r.get("decision"),
                "reason": r.get("reason"),
                "similarity": r.get("similarity", ""),
                "similar_paper": r.get("similar_paper", ""),
            })
    
    print(f"📄 Appended {len(results)} new results to {output_file}")
    print(f"📊 Total results now: {len(existing_ids) + len(results)}")
else:
    print("ℹ️ No new results to append.")