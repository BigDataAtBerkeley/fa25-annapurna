#!/bin/bash
# Setup OpenSearch Domain for Annapurna Pipeline

set -e

echo "=== Setting up OpenSearch Domain ==="

REGION=${AWS_REGION:-us-east-1}
DOMAIN_NAME="research-papers"
INDEX_NAME="research-papers-v3"

echo "Region: $REGION"
echo "Domain: $DOMAIN_NAME"
echo "Index: $INDEX_NAME"

echo ""
echo "WARNING: OpenSearch domain creation requires manual setup through AWS Console or CloudFormation."
echo ""
echo "Required Configuration:"
echo "  - Domain Name: $DOMAIN_NAME"
echo "  - Index Name: $INDEX_NAME"
echo "  - Enable KNN Search via Cosine Similarity (CANNOT be unconfigured once set)"
echo ""
echo "To create the OpenSearch domain manually:"
echo "  1. Go to AWS OpenSearch Service Console"
echo "  2. Create a new domain with name: $DOMAIN_NAME"
echo "  3. Configure the domain with appropriate instance types"
echo "  4. Set up access policies to allow Lambda functions to access it"
echo "  5. Once the domain is created, update the OPENSEARCH_ENDPOINT in Lambda environment variables"
echo ""
echo "To create the index with KNN enabled, use the following mapping:"
echo ""
cat <<'EOF'
{
  "settings": {
    "index": {
      "number_of_shards": 1,
      "knn": true,
      "knn.algo_param.ef_search": 100
    }
  },
  "mappings": {
    "properties": {
      "title": {"type": "text"},
      "title_normalized": {"type": "keyword"},
      "authors": {"type": "keyword"},
      "abstract": {"type": "text"},
      "abstract_embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib",
          "parameters": {"ef_construction": 128, "m": 24}
        }
      },
      "date": {"type": "date"},
      "s3_bucket": {"type": "keyword"},
      "s3_key": {"type": "keyword"},
      "sha_abstract": {"type": "keyword"},
      "decision": {"type": "keyword"},
      "rejected_by": {"type": "keyword"},
      "reason": {"type": "text"},
      "relevance": {"type": "keyword"},
      "ingested_at": {"type": "date"},
      "executed_on_trn": {"type": "boolean"}
    }
  }
}
EOF

echo ""
echo "After creating the domain, you can create the index using:"
echo "  curl -X PUT \"https://<OPENSEARCH_ENDPOINT>/$INDEX_NAME\" -H 'Content-Type: application/json' -d @index_mapping.json"
echo ""
echo "Or use the AWS CLI:"
echo "  aws opensearch create-index --domain-name $DOMAIN_NAME --index-name $INDEX_NAME --body file://index_mapping.json"
echo ""
echo "=== OpenSearch Setup Instructions Complete ==="
echo ""
echo "Note: This script does not create the OpenSearch domain automatically due to:"
echo "  1. Cost considerations (OpenSearch domains can be expensive)"
echo "  2. Complex configuration requirements"
echo "  3. Need for careful planning of instance types and capacity"
echo ""
echo "Please set up the OpenSearch domain manually and update Lambda environment variables accordingly."

