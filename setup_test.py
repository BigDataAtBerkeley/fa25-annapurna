#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end script (copy-paste ready):

- Connects to OpenSearch with longer timeouts & retries
- Randomly samples up to 100 documents (no .count() / no deep pagination)
- Sends docs to an SQS "send" queue
- Polls an SQS "results" queue for decisions
- Clusters REJECTED papers by TOPIC (from title+abstract) using TF-IDF + KMeans
  * uses positional indexing so TF-IDF slices never go out-of-range
- Writes paper_results.csv including cluster_id / cluster_name / cluster_top_terms
- Writes clusters_summary.csv and prints a compact cluster summary

Requires:
  pip install pandas numpy scikit-learn python-dotenv opensearch-py boto3
"""

import os
import re
import json
import time
import random
import csv
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================ CONFIG / CLIENTS ============================

# Load env
load_dotenv()

REGION = os.getenv("AWS_REGION")
HOST = os.getenv("OPENSEARCH_ENDPOINT")
INDEX = os.getenv("OPENSEARCH_INDEX")
SEND_QUEUE_URL = os.getenv("TEST_QUEUE_URL")
RECV_QUEUE_URL = os.getenv("RESULTS_QUEUE_URL")

if not all([REGION, HOST, INDEX, SEND_QUEUE_URL, RECV_QUEUE_URL]):
    raise EnvironmentError(
        "Missing one or more env vars: AWS_REGION, OPENSEARCH_ENDPOINT, OPENSEARCH_INDEX, TEST_QUEUE_URL, RESULTS_QUEUE_URL"
    )

# AWS auth
session = boto3.Session(region_name=REGION)
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
sample_size = min(250, total_docs)
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
auth = AWSV4SignerAuth(credentials, REGION, "es")

# Tougher OpenSearch client (longer timeout + retries)
def make_os_client(host: str, auth: AWSV4SignerAuth) -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": host.replace("https://", "").replace("http://", ""), "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,           # socket read timeout
        max_retries=5,
        retry_on_timeout=True,
    )

os_client = make_os_client(HOST, auth)

# SQS client
sqs = boto3.client("sqs", region_name=REGION)

# Health ping (non-fatal)
try:
    info = os_client.info(request_timeout=30)
    print(f"🔌 OpenSearch OK — version: {info.get('version',{}).get('number','?')}")
except Exception as e:
    print(f"⚠️ OS health/ping issue: {e} (continuing with longer timeouts)")


# ============================ UTILITIES ============================

def safe_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """
    Returns df[col] as a string Series if present; otherwise a length-matched Series of defaults.
    Prevents .get(...)=str cases causing 'str' has no attribute 'fillna' errors.
    """
    if col in df.columns:
        s = df[col]
        if not isinstance(s, pd.Series):
            s = pd.Series([s] * len(df), index=df.index)
        return s.astype("string").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype="string")


# ============================ SAMPLING (NO COUNT) ============================

def fetch_random_batch(size=50, seed=None, timeout=45) -> Dict[str, Any]:
    """Use function_score.random_score to avoid deep pagination and count()."""
    body = {
        "size": size,
        "_source": ["title", "name", "abstract", "authors"],  # limit payload
        "query": {
            "function_score": {
                "query": {"match_all": {}},
                "random_score": ({"seed": seed} if seed is not None else {}),
            }
        },
    }
    return os_client.search(index=INDEX, body=body, request_timeout=timeout)


def sample_docs(sample_size: int = 100, max_tries: int = 20) -> List[Dict[str, Any]]:
    print(f"\n📦 Sampling up to {sample_size} random documents (no count)...")
    seen = set()
    docs: List[Dict[str, Any]] = []
    tries = 0
    while len(docs) < sample_size and tries < max_tries:
        tries += 1
        try:
            res = fetch_random_batch(
                size=min(200, sample_size - len(docs)),
                seed=random.randint(0, 1_000_000),
                timeout=45,
            )
            hits = res.get("hits", {}).get("hits", [])
            for h in hits:
                _id = h.get("_id")
                if not _id or _id in seen:
                    continue
                seen.add(_id)
                docs.append(h)
            print(f"  … got {len(docs)}/{sample_size} so far (try {tries}/{max_tries})")
        except Exception as e:
            print(f"⚠️ random batch failed (try {tries}): {e}")
            time.sleep(2)
    if not docs:
        raise RuntimeError("Could not fetch any random documents from OpenSearch.")
    print(f"✅ Sampled {len(docs)} documents.")
    return docs


# ============================ SEND TO SQS ============================

def send_docs_to_queue(docs: List[Dict[str, Any]]) -> List[str]:
    sent_ids = []
    for i, doc in enumerate(docs, start=1):
        try:
            src = doc.get("_source", {}) or {}
            doc_id = doc.get("_id")

            message = {
                "paper_id": doc_id,
                "title": src.get("title") or src.get("name") or "Untitled",
                "abstract": src.get("abstract", "No abstract available"),
                "authors": src.get("authors", []),
            }

            sqs.send_message(
                QueueUrl=SEND_QUEUE_URL,
                MessageBody=json.dumps(message),
                MessageAttributes={
                    "PaperName": {"StringValue": message["title"], "DataType": "String"},
                },
            )
            sent_ids.append(doc_id)
            if i % 10 == 0 or i == len(docs):
                print(f"✅ sent {i}/{len(docs)}")
            time.sleep(0.15)  # be gentle
        except Exception as e:
            print(f"❌ Error sending id={doc.get('_id')}: {e}")
    return sent_ids


# ============================ POLL RESULTS ============================

def poll_results(expected: int, timeout_seconds: int = 600) -> List[Dict[str, Any]]:
    print("\n🚦 Waiting for results...")
    results: List[Dict[str, Any]] = []
    received_ids = set()
    start_time = time.time()

    while len(received_ids) < expected and (time.time() - start_time) < timeout_seconds:
        resp = sqs.receive_message(
            QueueUrl=RECV_QUEUE_URL,
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
                paper_id = payload.get("paper_id") or f"msg_{len(results) + 1}"

                if paper_id not in received_ids:
                    results.append(payload)
                    received_ids.add(paper_id)

                    count = len(received_ids)
                    if count == 1:
                        print("📥 Received first paper result!")
                    elif count % 10 == 0:
                        print(f"📈 Received {count} results so far...")
                    if count == expected:
                        print("🎯 Received all expected results!")

                # delete processed message
                sqs.delete_message(
                    QueueUrl=RECV_QUEUE_URL,
                    ReceiptHandle=msg["ReceiptHandle"],
                )
            except Exception as e:
                print(f"⚠️ Error parsing message: {e}")

    print(f"\n✅ Done! Received {len(received_ids)}/{expected} results.")
    return results


# ============================ CLUSTER REJECTIONS (TOPIC LABELS, SAFE INDEXING) ============================

def cluster_and_label(recs: List[Dict[str, Any]]):
    """
    Cluster rejected papers and label clusters by topic (from title+abstract),
    using TF-IDF + KMeans. Uses positional indices to slice the TF–IDF matrix safely.
    """
    if not recs:
        return {}, pd.DataFrame()

    df_local = pd.DataFrame(recs).copy()
    df_local["decision"] = safe_series(df_local, "decision", "")
    df_local["paper_id"] = safe_series(df_local, "paper_id", "")

    mask_reject = df_local["decision"].str.lower().eq("reject")
    if not mask_reject.any():
        print("ℹ️ No rejected papers to cluster.")
        return {}, pd.DataFrame()

    rej = df_local[mask_reject].copy()
    # IMPORTANT: make row labels contiguous so we can slice X safely
    rej.reset_index(drop=True, inplace=True)

    title_s    = safe_series(rej, "title", "")
    abstract_s = safe_series(rej, "abstract", "")

    # Compose topic text (ignore 'reason' since it’s mostly “out of scope”)
    rej["text"] = (
        title_s.str.strip().fillna("") + " | " +
        abstract_s.str.strip().fillna("")
    ).str.replace(r"\s+", " ", regex=True)

    n = len(rej)

    # Build TF-IDF (even for tiny n so we can extract topic terms)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = tfidf.fit_transform(rej["text"])

    def top_terms_for_rows(row_idx, topn=6, prefer=4):
        """Return readable topic terms prioritizing bigrams. row_idx is a numpy array of positions."""
        if isinstance(row_idx, slice):
            mat = X[row_idx]
        else:
            row_idx = np.asarray(row_idx, dtype=int)
            if row_idx.size == 0:
                return []
            mat = X[row_idx]
        row_mean = mat.mean(axis=0).A1
        idxs = row_mean.argsort()[-(topn*4):][::-1]  # take plenty, then filter
        terms = [tfidf.get_feature_names_out()[i] for i in idxs]
        bigrams, unigrams, seen = [], [], set()
        for t in terms:
            cleaned = t.strip().lower().replace(" llm ", " llm").replace("  ", " ")
            if cleaned in seen:
                continue
            seen.add(cleaned)
            (bigrams if " " in cleaned else unigrams).append(cleaned)
        out = bigrams[:prefer] + unigrams[:max(0, topn - len(bigrams[:prefer]))]
        return out

    if n < 3:
        label = " • ".join(top_terms_for_rows(slice(None), topn=4, prefer=3)) or "misc"
        rej["cluster_id"] = 0
        rej["cluster_name"] = label
        cluster_map = {
            pid: {"cluster_id": 0, "cluster_name": label, "cluster_top_terms": label}
            for pid in rej["paper_id"]
        }
        summary_df = pd.DataFrame([{"cluster_id": 0, "cluster_name": label,
                                    "cluster_top_terms": label, "n": n}])
        return cluster_map, summary_df

    # Choose k by silhouette in [2..min(8, n//3)]
    k_hi = max(2, min(8, max(2, n // 3)))
    ks = list(range(2, max(k_hi, 2) + 1))
    best_k, best_s = 2, -1
    for k in ks:
        try:
            km_tmp = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
            s = silhouette_score(X, km_tmp.labels_) if X.shape[0] > k else -1
            if s > best_s:
                best_k, best_s = k, s
        except Exception:
            continue

    km = KMeans(n_clusters=max(best_k, 2), n_init="auto", random_state=42).fit(X)
    rej["cluster_id"] = km.labels_

    def topic_label(cid):
        # use positional mask
        pos = np.where(rej["cluster_id"].to_numpy() == cid)[0]
        terms = top_terms_for_rows(pos, topn=5, prefer=4)
        return " • ".join(terms) if terms else f"cluster_{cid}"

    rows = []
    cluster_map = {}
    for cid in sorted(rej["cluster_id"].unique()):
        pos = np.where(rej["cluster_id"].to_numpy() == cid)[0]
        label = topic_label(cid)
        rej.loc[pos, "cluster_name"] = label
        # Map to each paper (use iloc with positional rows)
        for pid in rej.iloc[pos]["paper_id"]:
            cluster_map[pid] = {"cluster_id": int(cid), "cluster_name": label, "cluster_top_terms": label}
        examples = rej.iloc[pos][["title"]].head(3).to_dict("records")
        rows.append({
            "cluster_id": int(cid),
            "cluster_name": label,
            "cluster_top_terms": label,
            "n": int(len(pos)),
            "example_1": json.dumps(examples[0], ensure_ascii=False) if len(examples) > 0 else "",
            "example_2": json.dumps(examples[1], ensure_ascii=False) if len(examples) > 1 else "",
            "example_3": json.dumps(examples[2], ensure_ascii=False) if len(examples) > 2 else "",
        })

    summary_df = pd.DataFrame(rows).sort_values(["n", "cluster_id"], ascending=[False, True])
    return cluster_map, summary_df


# ============================ MAIN ============================

def main():
    # 1) sample docs (no count) & send to SQS
    docs = sample_docs(sample_size=100, max_tries=20)
    send_docs_to_queue(docs)

    # 2) poll results
    results = poll_results(expected=len(docs), timeout_seconds=600)

    # 3) cluster rejects (topic labels + safe indexing)
    cluster_map, summary_df = cluster_and_label(results)
    if not summary_df.empty:
        summary_df.to_csv("clusters_summary.csv", index=False)
        print("🧭 Wrote clusters_summary.csv")

    # 4) write master CSV (with clusters)
    output_file = "paper_results.csv"
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "paper_id",
                "title",
                "decision",
                "relevance",
                "novelty",
                "trainium_compatibility",
                "reason",
                "similarity",
                "similar_paper",
                "cluster_id",
                "cluster_name",
                "cluster_top_terms",   # topic-style label
            ],
        )
        writer.writeheader()
        for r in results:
            info = cluster_map.get(r.get("paper_id"), {})
            writer.writerow({
                "paper_id": r.get("paper_id"),
                "title": r.get("title"),
                "decision": r.get("decision"),
                "relevance": r.get("relevance"),
                "novelty": r.get("novelty"),
                "trainium_compatibility": r.get("trainium_compatibility"),
                "reason": r.get("reason") or r.get("claude_reason"),
                "similarity": r.get("max_similarity", ""),
                "similar_paper": r.get("most_similar_paper", ""),
                "cluster_id": info.get("cluster_id", ""),
                "cluster_name": info.get("cluster_name", ""),
                "cluster_top_terms": info.get("cluster_top_terms", ""),
            })
    print(f"📄 Results (with clusters) saved to {output_file}")

    # 5) console summary
    if cluster_map:
        df_all = pd.read_csv(output_file)
        rej_summary = (
            df_all[df_all["decision"].str.lower() == "reject"]
            .groupby(["cluster_id", "cluster_name", "cluster_top_terms"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print("\n🔎 Rejection clusters (topics):")
        for _, row in rej_summary.iterrows():
            print(f"  • [{int(row['cluster_id'])}] {row['cluster_top_terms']}: {int(row['count'])} papers")
    else:
        print("\nℹ️ No rejected papers were clustered (possibly none were rejected).")


if __name__ == "__main__":
    main()
