## JUST RUN `python check_opensearch.py` TO CHECK OPENSEARCH DOCS

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import json
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()  

# === Configuration ===
region = os.getenv("AWS_REGION")
host = os.getenv("OPENSEARCH_ENDPOINT")
index_name = os.getenv("OPENSEARCH_INDEX")

if not region or not host or not index_name:
    raise EnvironmentError(
        "Missing one or more environment variables: AWS_REGION, OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX"
    )

# === Auth setup ===
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

# === 1. Cluster health ===
print("\n✅ Cluster Health:")
health = os_client.cluster.health()
print(json.dumps(health, indent=2))

# === 2. List indices ===
print("\nIndices:")
indices = os_client.cat.indices(format="json")
for idx in indices:
    print(f"- {idx['index']} ({idx['docs.count']} docs, status={idx['health']})")

# === 3. Fetch and print sample documents ===
print(f"\nDocuments from '{index_name}':")
try:
    res = os_client.search(index=index_name, body={"query": {"match_all": {}}, "size": 5})
    hits = res["hits"]["hits"]
    if not hits:
        print("No documents found in this index yet.")
    else:
        for i, hit in enumerate(hits, 1):
            src = hit["_source"]
            print(f"\n--- Document #{i} ---")
            print(f"Title: {src.get('title', 'N/A')}")
            print(f"Decision: {src.get('decision', 'N/A')}")
            print(f"Reason: {src.get('reason', 'N/A')}")
            print(f"Authors: {src.get('authors', [])}")
            print(f"Date: {src.get('date', 'N/A')}")
            print(f"S3 Key: {src.get('s3_key', 'N/A')}")
except Exception as e:
    print(f"⚠️ Error while fetching documents: {e}")
