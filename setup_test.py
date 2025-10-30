import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import json
import os
import random
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

# === Configuration ===
region = os.getenv("AWS_REGION")
host = os.getenv("OPENSEARCH_ENDPOINT")
index_name = os.getenv("OPENSEARCH_INDEX")
sqs_url = os.getenv("TEST_QUEUE_URL")  

if not region or not host or not index_name or not sqs_url:
    raise EnvironmentError(
        "Missing one or more required env vars: AWS_REGION, OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX, SQS_QUEUE_URL."
    )

# === AWS Auth ===
session = boto3.Session(region_name=region)
credentials = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(credentials, region, "es")

# === OpenSearch client ===
os_client = OpenSearch(
    hosts=[{"host": host.replace("https://", "").replace("http://", ""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# === SQS client ===
sqs = boto3.client("sqs", region_name=region)

# === 1. Get document count ===
count_response = os_client.count(index=index_name)
total_docs = count_response.get("count", 0)

if total_docs == 0:
    raise ValueError(f"No documents found in index '{index_name}'")

print(f"\n✅ Found {total_docs} total documents in '{index_name}'")

# === 2. Randomly sample 100 unique offsets ===
sample_size = min(100, total_docs)
random_offsets = random.sample(range(total_docs), sample_size)

print(f"📦 Sampling {sample_size} random documents...")

# === 3. Fetch and send sampled documents ===
for i, offset in enumerate(random_offsets, start=1):
    try:
        # Fetch single document by offset
        query = {"from": offset, "size": 1, "query": {"match_all": {}}}
        res = os_client.search(index=index_name, body=query)

        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            print(f"⚠️ No hit returned at offset {offset}")
            continue

        doc = hits[0]
        src = doc.get("_source", {})

        # Extract metadata fields
        message = {
            "paper_name": src.get("title") or src.get("name") or "Untitled",
            "s3_bucket": src.get("s3_bucket", "N/A"),
            "s3_link": src.get("s3_link", "N/A"),
            "abstract": src.get("abstract", "No abstract available"),
            "authors": src.get("authors", []),
        }

        # Send message to SQS
        sqs.send_message(
            QueueUrl=sqs_url,
            MessageBody=json.dumps(message),
            MessageAttributes={
                "PaperName": {"StringValue": message["paper_name"], "DataType": "String"},
                "S3Bucket": {"StringValue": str(message["s3_bucket"]), "DataType": "String"},
                "S3Link": {"StringValue": str(message["s3_link"]), "DataType": "String"},
                "AuthorsCount": {
                    "StringValue": str(len(message["authors"])),
                    "DataType": "Number",
                },
            },
        )

        print(f"✅ [{i}/{sample_size}] Sent '{message['paper_name']}' to SQS")

    except Exception as e:
        print(f"❌ Error at offset {offset}: {e}")

print("\n🎉 Done sending sampled documents to SQS.")
