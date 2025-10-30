# cluster.py
import json, re, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- helpers ----------
def load_snapshot(path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data.get("documents", [])
    return docs, data

def doc_to_text(d: Dict[str, Any], fields: List[str]) -> str:
    """Concatenate common text fields; fallback to any string fields."""
    chunks = [str(d.get(f, "")).strip() for f in fields if d.get(f)]
    if not chunks:
        chunks = [str(v) for k, v in d.items() if isinstance(v, str)]
    text = " \n".join(chunks)
    # light cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text

def vectorize_corpus(corpus: List[str]) -> Tuple[TfidfVectorizer, Any]:
    vec = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    X = vec.fit_transform(corpus)
    return vec, X

def top_terms_per_cluster(km: KMeans, vec: TfidfVectorizer, top_k: int = 10) -> List[List[str]]:
    # centers shape: (n_clusters, n_features)
    centers = km.cluster_centers_
    vocab = vec.get_feature_names_out()
    top_terms = []
    for c in centers:
        idx = c.argsort()[::-1][:top_k]
        top_terms.append([vocab[i] for i in idx])
    return top_terms

# ---------- main clustering function ----------
def cluster_articles(
    json_path: str | Path = "cluster_input.json",
    n_clusters: int = 8,
    fields_for_text: List[str] = ("title", "abstract", "summary", "content"),
    out_prefix: str | Path = "clusters"
):
    # 1) load docs
    docs, meta = load_snapshot(json_path)
    if not docs:
        print("No documents in snapshot.")
        return

    ids = [d.get("_id") or d.get("id") or f"doc_{i}" for i, d in enumerate(docs)]
    titles = [d.get("title") or d.get("title_normalized") or ids[i] for i, d in enumerate(docs)]
    corpus = [doc_to_text(d, list(fields_for_text)) for d in docs]

    # 2) TF-IDF
    vec, X = vectorize_corpus(corpus)

    # 3) K-Means
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    # 4) (optional) quality metric
    try:
        sil = silhouette_score(X, labels) if n_clusters > 1 else None
    except Exception:
        sil = None

    # 5) summarize clusters
    terms = top_terms_per_cluster(km, vec, top_k=10)
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)

    print(f"\nClustered {len(docs)} docs from index '{meta.get('index')}' "
          f"into {n_clusters} clusters.")
    if sil is not None:
        print(f"Silhouette score (TF-IDF space): {sil:.3f}")

    for cid in sorted(clusters):
        members = clusters[cid]
        sample_titles = [titles[i] for i in members[:8]]
        print(f"\n— Cluster {cid} ({len(members)} docs)")
        print("   top terms:", ", ".join(terms[cid]))
        for t in sample_titles:
            print(f"   • {t}")

    # 6) write labeled outputs
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # JSON lines with label
    jl_path = out_prefix.with_suffix(".jsonl")
    with jl_path.open("w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            row = {"_id": ids[i], "title": titles[i], "cluster": int(labels[i])}
            row["preview"] = corpus[i][:240]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Simple CSV
    csv_path = out_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["_id", "title", "cluster"])
        for i in range(len(docs)):
            w.writerow([ids[i], titles[i], int(labels[i])])

    print(f"\nSaved:\n  {jl_path}\n  {csv_path}")

if __name__ == "__main__":
    # quick run with defaults
    cluster_articles(
        json_path="cluster_input.json",
        n_clusters=8,
        fields_for_text=("title", "abstract", "summary", "content"),
        out_prefix="clustering/result_clusters"
    )
