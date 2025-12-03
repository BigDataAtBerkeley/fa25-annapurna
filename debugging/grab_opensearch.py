# check_opensearch.py  (or grab_opensearch.py)

import os, json, argparse
from pathlib import Path

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from dotenv import load_dotenv, find_dotenv

# === Load environment variables ===
load_dotenv(find_dotenv())

region = os.getenv("AWS_REGION")
host = os.getenv("OPENSEARCH_ENDPOINT")
index_name = os.getenv("OPENSEARCH_INDEX")

if not region or not host or not index_name:
    raise EnvironmentError(
        "Missing one or more env vars: AWS_REGION, OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX"
    )

# === Auth + client ===
session = boto3.Session(region_name=region)
credentials = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(credentials, region, "es")

os_client = OpenSearch(
    hosts=[{"host": host.replace("https://", "").replace("http://", ""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

# === CLI args ===
parser = argparse.ArgumentParser(
    description="Check OpenSearch and optionally dump docs for clustering."
)
parser.add_argument(
    "--json-out",
    default="cluster_input.json",
    help="Path to write a JSON snapshot (cluster health, indices, docs)",
)
parser.add_argument(
    "--size", type=int, default=30, help="How many docs to fetch from index"
)
args = parser.parse_args()

# === 1. Cluster health ===
print("\n✅ Cluster Health:")
health = os_client.cluster.health()
print(json.dumps(health, indent=2))

# === 2. List indices ===
print("\nIndices:")
indices = os_client.cat.indices(format="json")
for idx in indices:
    print(f"- {idx['index']} ({idx.get('docs.count', '?')} docs, status={idx.get('health', '?')})")

# === 3. Fetch and print sample documents ===
print(f"\nDocuments from '{index_name}':")
docs_for_dump: list[dict] = []  # <-- define it here
try:
    res = os_client.search(
        index=index_name,
        body={"query": {"match_all": {}}, "size": args.size},
    )
    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        print("No documents found in this index yet.")
    else:
        for i, hit in enumerate(hits, 1):
            _id = hit.get("_id", "N/A")
            src = hit.get("_source", {})
            docs_for_dump.append({"_id": _id, **src})  # <-- collect for JSON

            print("\n" + "=" * 80)
            print(f"Document #{i} (ID: {_id})")
            print("=" * 80)
            for field, value in sorted(src.items()):
                if isinstance(value, list):
                    value_str = f"{value[:3]}... ({len(value)} items)" if len(value) > 3 else str(value)
                elif isinstance(value, str) and len(value) > 200 and field != "abstract":
                    value_str = value[:200] + "..."
                else:
                    value_str = str(value)
                print(f"  {field}: {value_str}")
except Exception as e:
    print(f"⚠️ Error while fetching documents: {e}")

# === 4. Write a JSON snapshot for cluster.py ===
snapshot = {
    "region": region,
    "endpoint": host,
    "index": index_name,
    "cluster_health": health,
    "indices": indices,
    "documents": docs_for_dump,
}

out_path = Path(args.json_out).expanduser().resolve()
out_path.parent.mkdir(parents=True, exist_ok=True)

try:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Wrote snapshot for clustering → {out_path} (docs: {len(docs_for_dump)})")
    print(f"(cwd: {os.getcwd()})")
except Exception as e:
    print(f"⚠️ Failed to write JSON to {out_path}: {e}")
