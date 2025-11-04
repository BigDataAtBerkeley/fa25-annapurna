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

# === 2. Randomly select 100 ===
sample_size = min(1, total_docs)
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

print("\n🚀 All 100 documents sent. Now waiting for results...\n")

# === 4. Poll results queue ===
results = []
received_ids = set()
start_time = time.time()
timeout_seconds = 600  # 10 min timeout

while len(received_ids) < sample_size and (time.time() - start_time) < timeout_seconds:
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
            # Depending on Lambda setup, this might be wrapped in "responsePayload"
            payload = body.get("responsePayload") or body
            paper_id = payload.get("paper_id") or f"msg_{len(results)+1}"

            if paper_id not in received_ids:
                results.append(payload)
                received_ids.add(paper_id)

                count = len(received_ids)
                if count == 1:
                    print("📥 Received first paper result!")
                elif count % 5 == 0:
                    print(f"📈 Received {count} results so far...")
                if count == sample_size:
                    print("🎯 Received all 100 results!")

            # Delete processed message
            sqs.delete_message(
                QueueUrl=receive_queue_url,
                ReceiptHandle=msg["ReceiptHandle"]
            )

        except Exception as e:
            print(f"⚠️ Error parsing message: {e}")

print(f"\n✅ Done! Received {len(received_ids)}/{sample_size} results.")

# === 5. Save results to CSV ===
output_file = "paper_results.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["paper_id", "title", "decision", "reason", "similarity", "similar_paper"])
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
