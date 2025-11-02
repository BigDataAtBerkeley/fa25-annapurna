import boto3
import json
import csv
import os
import time
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

receive_queue_url = os.getenv("RESULTS_QUEUE_URL")  
if not receive_queue_url:
    raise EnvironmentError("Missing env var: RESULTS_QUEUE_URL")

# === AWS SQS Client ===
region = os.getenv("AWS_REGION", "us-east-1")
sqs = boto3.client("sqs", region_name=region)

# === Poll results queue ===
results = []
received_ids = set()
start_time = time.time()
timeout_seconds = 600  # 10 min timeout

print("⏳ Waiting for results from SQS...")

while (time.time() - start_time) < timeout_seconds:
    resp = sqs.receive_message(
        QueueUrl=receive_queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=10,
    )

    messages = resp.get("Messages", [])
    if not messages:
        continue

    for msg in messages:
        try:
            body = json.loads(msg["Body"])
            payload = body.get("responsePayload") or body
            paper_id = payload.get("paper_id") or f"msg_{len(results)+1}"

            if paper_id not in received_ids:
                results.append(payload)
                received_ids.add(paper_id)
                print(f"📥 Received result for '{payload.get('title', paper_id)}' ({len(received_ids)})")

            # Delete processed message
            sqs.delete_message(
                QueueUrl=receive_queue_url,
                ReceiptHandle=msg["ReceiptHandle"]
            )

        except Exception as e:
            print(f"⚠️ Error parsing message: {e}")

print(f"\n✅ Done! Received {len(received_ids)} results.")

# === Save results to CSV ===
output_file = "paper_results.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    fieldnames = ["paper_id", "title", "decision", "reason", "similarity", "similar_paper"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
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

print(f"📄 Results saved to {output_file}")
