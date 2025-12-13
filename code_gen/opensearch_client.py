"""
OpenSearch client for retrieving research papers and metadata
"""

import os
import logging
from typing import Dict, List, Optional, Any

import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

load_dotenv()

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client for paper retrieval"""

    def __init__(self):
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        self.index = os.getenv("OPENSEARCH_INDEX", "research-papers-v3")

        if not self.endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT must be set")

        session = boto3.Session(region_name=self.aws_region)
        creds = session.get_credentials().get_frozen_credentials()
        auth = AWSV4SignerAuth(creds, self.aws_region, "es")

        self.client = OpenSearch(
            hosts=[{
                "host": self.endpoint.replace("https://", "").replace("http://", ""),
                "port": 443
            }],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )

        logger.info(f"OpenSearch initialized (index={self.index})")


    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.client.get(index=self.index, id=paper_id)["_source"]
        except Exception as e:
            logger.error(f"Failed to retrieve paper {paper_id}: {e}")
            return None

    def search_papers(self, query: Dict[str, Any], size: int = 10) -> List[Dict[str, Any]]:
        """Run a raw OpenSearch query"""
        try:
            resp = self.client.search(
                index=self.index,
                body={"query": query, "size": size},
            )
            results = []
            for hit in resp.get("hits", {}).get("hits", []):
                doc = hit["_source"]
                doc["_id"] = hit["_id"]
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"Paper search failed: {e}")
            return []


    def get_papers_by_title(self, title: str, size: int = 5) -> List[Dict[str, Any]]:
        return self.search_papers({"match": {"title": title}}, size)

    def get_papers_by_author(self, author: str, size: int = 5) -> List[Dict[str, Any]]:
        return self.search_papers({"match": {"authors": author}}, size)

    def get_papers_by_abstract(self, keywords: str, size: int = 5) -> List[Dict[str, Any]]:
        return self.search_papers({"match": {"abstract": keywords}}, size)

    def get_recent_papers(self, days: int = 30, size: int = 10) -> List[Dict[str, Any]]:
        return self.search_papers(
            {"range": {"ingested_at": {"gte": f"now-{days}d"}}},
            size,
        )

    def get_random_papers(self, size: int = 10) -> List[Dict[str, Any]]:
        return self.search_papers(
            {
                "function_score": {
                    "query": {"match_all": {}},
                    "random_score": {},
                    "boost_mode": "replace",
                }
            },
            size,
        )

    def get_all_papers(self, size: int = 50) -> List[Dict[str, Any]]:
        return self.search_papers({"match_all": {}}, size)

    def get_paper_content(self, paper: Dict[str, Any]) -> Optional[str]:
        """Fetch non-PDF paper content from S3"""
        bucket = paper.get("s3_bucket")
        key = paper.get("s3_key")

        if not bucket or not key:
            return None
        if key.lower().endswith(".pdf"):
            return None

        try:
            s3 = boto3.client("s3", region_name=self.aws_region)
            raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

            for enc in ("utf-8", "utf-16", "latin-1", "cp1252", "iso-8859-1"):
                try:
                    return raw.decode(enc)
                except UnicodeError:
                    continue

            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"S3 content fetch failed: {e}")
            return None

    def get_paper_pdf_bytes(self, paper: Dict[str, Any]) -> Optional[bytes]:
        """Fetch raw PDF bytes from S3"""
        bucket = paper.get("s3_bucket")
        key = paper.get("s3_key")

        if not bucket or not key:
            return None

        try:
            s3 = boto3.client("s3", region_name=self.aws_region)
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:
            logger.error(f"S3 PDF fetch failed: {e}")
            return None

    def is_pdf_paper(self, paper: Dict[str, Any]) -> bool:
        return paper.get("s3_key", "").lower().endswith(".pdf")

    def get_paper_summary(self, paper: Dict[str, Any]) -> str:
        return (
            f"Paper Title: {paper.get('title', 'Unknown')}\n"
            f"Authors: {', '.join(paper.get('authors', []))}\n"
            f"Date: {paper.get('date', 'Unknown')}\n"
            f"Abstract: {paper.get('abstract', 'N/A')}"
        )


    def search_similar_papers_by_abstract(
        self,
        paper_id: str,
        exclude_id: Optional[str] = None,
        size: int = 5,
    ) -> List[Dict[str, Any]]:
        """kNN search using stored abstract embeddings to find similar papers in OpenSearch"""
        paper = self.get_paper_by_id(paper_id)
        if not paper or "abstract_embedding" not in paper:
            return []

        try:
            embedding = [float(x) for x in paper["abstract_embedding"]]
        except Exception:
            return []

        query_k = size + 1 if exclude_id else size
        query = {
            "knn": {
                "abstract_embedding": {
                    "vector": embedding,
                    "k": query_k,
                }
            }
        }

        try:
            resp = self.client.search(
                index=self.index,
                body={"query": query, "size": query_k},
            )
        except Exception:
            query = {
                "knn": {
                    "field": "abstract_embedding",
                    "query_vector": embedding,
                    "k": query_k,
                }
            }
            resp = self.client.search(
                index=self.index,
                body={"query": query, "size": query_k},
            )

        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            if exclude_id and hit["_id"] == exclude_id:
                continue
            doc = hit["_source"]
            doc["_id"] = hit["_id"]
            doc["similarity_score"] = hit.get("_score")
            results.append(doc)
            if len(results) >= size:
                break

        return results