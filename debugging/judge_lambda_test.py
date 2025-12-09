import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import json
import os
import random
import csv
import time
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

# === Configuration ===
region = os.getenv("AWS_REGION")
host = os.getenv("OPENSEARCH_ENDPOINT")
index_name = os.getenv("OPENSEARCH_INDEX")
send_queue_url = os.getenv("TEST_QUEUE_URL")
receive_queue_url = os.getenv("RESULTS_QUEUE_URL")  

if not region or not host or not index_name or not send_queue_url or not receive_queue_url:
    raise EnvironmentError(
        "Missing one or more env vars: AWS_REGION, OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX, TEST_QUEUE_URL, RESULTS_QUEUE_URL"
    )

# === AWS Clients ===
session = boto3.Session(region_name=region)
credentials = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(credentials, region, "es")

os_client = OpenSearch(
    hosts=[{"host": host.replace("https://", "").replace("http://", ""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

sqs = boto3.client("sqs", region_name=region)

# === 1. Get total documents ===
count_response = os_client.count(index=index_name)
total_docs = count_response.get("count", 0)
if total_docs == 0:
    raise ValueError(f"No documents found in index '{index_name}'")

print(f"\n✅ Found {total_docs} total documents in '{index_name}'")

# === 2. Randomly select 100–150 ===
sample_size = min(200, total_docs)
random_offsets = random.sample(range(total_docs), sample_size)
print(f"📦 Sampling {sample_size} random documents...")

# === 3. Send to SQS ===
sent_ids = []
for i, offset in enumerate(random_offsets, start=1):
    try:
        res = os_client.search(index=index_name, body={"from": offset, "size": 1, "query": {"match_all": {}}})
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            continue

        doc = hits[0]
        src = doc.get("_source", {})
        doc_id = doc.get("_id")

        message = {
            "paper_id": doc_id,
            "title": src.get("title") or src.get("name") or "Untitled",
            "abstract": src.get("abstract", "No abstract available"),
            "authors": src.get("authors", [])
        }

        sqs.send_message(
            QueueUrl=send_queue_url,
            MessageBody=json.dumps(message),
            MessageAttributes={
                "PaperName": {"StringValue": message["title"], "DataType": "String"},
            },
        )
        sent_ids.append(doc_id)
        print(f"✅ [{i}/{sample_size}] Sent '{message['title']}'")

    except Exception as e:
        print(f"❌ Error sending doc at offset {offset}: {e}")

print(f"\n🚀 All {sample_size} documents sent. Now waiting for results...\n")

# === 4. Prepare CSV for live updates ===
output_file = "paper_results_2.csv"
fieldnames = ["paper_id", "title", "decision", "reason", "similarity", "trainium_compatibility"]

file_exists = os.path.exists(output_file)
csv_file = open(output_file, mode="a", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

# Write header if new file
if not file_exists:
    writer.writeheader()
    csv_file.flush()

# === 5. Poll results queue and write live ===
results = []
received_ids = set()
start_time = time.time()
timeout_seconds = 1200  # 20 min timeout

try:
    while len(received_ids) < len(sent_ids) and (time.time() - start_time) < timeout_seconds:
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

                    # Write immediately to CSV
                    writer.writerow({
                        "paper_id": payload.get("paper_id"),
                        "title": payload.get("title"),
                        "decision": payload.get("decision"),
                        "reason": payload.get("reason"),
                        "similarity": payload.get("similarity", ""),
                        "trainium_compatibility": payload.get("trainium_compatibility", "")
                    })
                    csv_file.flush()

                    count = len(received_ids)
                    if count == 1:
                        print("📥 Received first paper result!")
                    elif count % 5 == 0:
                        print(f"📈 Received {count} results so far...")
                    if count == sample_size:
                        print(f"🎯 Received all {sample_size} results!")

                # Delete processed message
                sqs.delete_message(
                    QueueUrl=receive_queue_url,
                    ReceiptHandle=msg["ReceiptHandle"]
                )

            except Exception as e:
                print(f"⚠️ Error parsing message: {e}")

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user. Saving progress before exit...")

finally:
    csv_file.close()
    print(f"\n✅ Done! Received {len(received_ids)}/{sample_size} results saved to {output_file}")