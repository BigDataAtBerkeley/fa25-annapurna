# RAG Implementation Deployment Guide

This guide explains how to deploy the enhanced RAG-based paper deduplication system.

## Overview

The RAG implementation replaces the basic exact matching in the Judge Lambda with sophisticated vector-based similarity search using:

- **Amazon Titan Embeddings** for generating vector embeddings from paper abstracts
- **OpenSearch k-NN** for vector similarity search
- **Configurable similarity thresholds** for redundancy detection

## Prerequisites

### 1. AWS Services Access
- Amazon Bedrock with Titan Embeddings model access
- Amazon OpenSearch Service with k-NN support
- AWS Lambda execution permissions

### 2. Environment Variables

Add these to your `.env` file:

```bash
# Existing variables
OPENSEARCH_ENDPOINT=your-opensearch-endpoint
AWS_REGION=us-east-1
OPENSEARCH_INDEX=research-papers-v2

# New RAG-specific variables
EMBEDDINGS_MODEL_ID=amazon.titan-embed-text-v1
SIMILARITY_THRESHOLD=0.85
HIGH_SIMILARITY_THRESHOLD=0.95
MEDIUM_SIMILARITY_THRESHOLD=0.75
```

### 3. Model Access

Enable the Amazon Titan Embeddings model in Bedrock:

1. Go to Amazon Bedrock console
2. Navigate to "Model access"
3. Enable "Amazon Titan Embeddings G1 - Text"
4. Wait for "Access granted" status

## Deployment Steps

### 1. Update Dependencies

Install the updated requirements:

```bash
pip install -r code_gen/requirements.txt
```

### 2. Test RAG Implementation

Before deploying, test the implementation locally:

```bash
python test_rag_implementation.py
```

This will test:
- Embeddings generation
- OpenSearch vector index creation
- Paper redundancy detection
- Vector similarity search

### 3. Deploy Enhanced Judge Lambda

Replace the existing Judge Lambda with the RAG-enhanced version:

```bash
# Package the enhanced judge lambda
cd judge_lambda
zip -r judge_lambda_rag.zip lambda_function_rag.py

# Deploy to AWS Lambda
aws lambda update-function-code \
  --function-name PapersJudge \
  --zip-file fileb://judge_lambda_rag.zip
```

### 4. Update Lambda Environment Variables

Set the new environment variables for the Judge Lambda:

```bash
aws lambda update-function-configuration \
  --function-name PapersJudge \
  --environment Variables='{
    "OPENSEARCH_ENDPOINT":"your-endpoint",
    "OPENSEARCH_INDEX":"research-papers-v2",
    "AWS_REGION":"us-east-1",
    "SIMILARITY_THRESHOLD":"0.85",
    "EMBEDDINGS_MODEL_ID":"amazon.titan-embed-text-v1"
  }'
```

### 5. Deploy Enhanced Code Generator

Update the code generator with RAG capabilities:

```bash
cd code_gen
zip -r code_gen_rag.zip . -x "*.pyc" "__pycache__/*"

aws lambda update-function-code \
  --function-name PapersCodeGenerator \
  --zip-file fileb://code_gen_rag.zip
```

## Configuration

### Similarity Thresholds

Configure similarity thresholds based on your needs:

- **SIMILARITY_THRESHOLD=0.85**: Papers with 85%+ similarity are considered redundant
- **HIGH_SIMILARITY_THRESHOLD=0.95**: Near-duplicates (95%+ similarity)
- **MEDIUM_SIMILARITY_THRESHOLD=0.75**: Potential duplicates (75%+ similarity)

### OpenSearch k-NN Settings

The system automatically configures OpenSearch with optimal k-NN settings:

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100
    }
  },
  "mappings": {
    "properties": {
      "abstract_embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 128,
            "m": 24
          }
        }
      }
    }
  }
}
```

## Testing

### 1. Local Testing

Test the RAG implementation locally:

```bash
python test_rag_implementation.py
```

### 2. Integration Testing

Test the complete pipeline:

```bash
# 1. Scrape some papers
aws lambda invoke \
  --function-name PaperScraper_ICLR \
  --payload '{"MAX_PAPERS": "5"}' response.json

# 2. Wait for judge processing (automatic via SQS)
# Check logs to see RAG-based redundancy detection

# 3. Check OpenSearch for papers with embeddings
python check_opensearch.py
```

### 3. Monitor Performance

Monitor the enhanced system:

```bash
# Check Lambda logs
aws logs tail /aws/lambda/PapersJudge --since 10m --follow

# Check OpenSearch index
aws opensearch describe-domain --domain-name your-domain
```

## Migration from Old System

### 1. Backup Existing Data

Before migrating, backup your existing OpenSearch data:

```bash
# Export existing papers
python -c "
from code_gen.opensearch_client import OpenSearchClient
client = OpenSearchClient()
papers = client.get_all_papers(size=1000)
import json
with open('backup_papers.json', 'w') as f:
    json.dump(papers, f, indent=2)
"
```

### 2. Migrate Existing Papers

Add embeddings to existing papers:

```python
from code_gen.opensearch_client import OpenSearchClient

client = OpenSearchClient()
client.ensure_vector_index()  # Ensure vector index exists

# Get all existing papers
papers = client.get_all_papers(size=1000)

# Add embeddings to each paper
for paper in papers:
    paper_data = paper.copy()
    if 'abstract_embedding' not in paper_data:
        # Generate and add embedding
        client.index_paper_with_embedding(paper_data)
```

### 3. Verify Migration

Verify that existing papers now have embeddings:

```python
from code_gen.opensearch_client import OpenSearchClient

client = OpenSearchClient()
papers = client.get_all_papers(size=10)

for paper in papers:
    has_embedding = 'abstract_embedding' in paper
    print(f"Paper: {paper.get('title', 'Unknown')}")
    print(f"Has embedding: {has_embedding}")
    if has_embedding:
        print(f"Embedding dimension: {len(paper['abstract_embedding'])}")
    print("-" * 50)
```

## Performance Considerations

### 1. Embeddings Generation

- Titan Embeddings generates 1536-dimensional vectors
- Generation time: ~1-2 seconds per abstract
- Cost: ~$0.0001 per 1K tokens

### 2. Vector Search Performance

- k-NN search is optimized for similarity queries
- Typical search time: 50-200ms
- Index size: ~6KB per paper (1536 dimensions × 4 bytes)

### 3. Memory Usage

- Lambda memory recommendation: 1024MB+ for embeddings generation
- OpenSearch storage: ~6KB additional per paper

## Troubleshooting

### Common Issues

1. **Embeddings generation fails**
   - Check Bedrock model access
   - Verify environment variables
   - Check Lambda permissions

2. **Vector search returns no results**
   - Verify OpenSearch index has vector mapping
   - Check if papers have embeddings
   - Verify similarity threshold settings

3. **High similarity scores for different papers**
   - Adjust similarity threshold
   - Check abstract quality
   - Consider preprocessing abstracts

### Debug Commands

```bash
# Check OpenSearch index mapping
curl -X GET "your-opensearch-endpoint/research-papers-v2/_mapping"

# Check if papers have embeddings
curl -X GET "your-opensearch-endpoint/research-papers-v2/_search" \
  -H "Content-Type: application/json" \
  -d '{"query": {"exists": {"field": "abstract_embedding"}}}'

# Test embeddings generation
python -c "
from code_gen.embeddings_client import EmbeddingsClient
client = EmbeddingsClient()
embedding = client.generate_embedding('Test abstract')
print(f'Generated embedding with {len(embedding)} dimensions')
"
```

## Cost Estimation

### Per 100 Papers

- **Embeddings Generation**: ~$0.50 (Titan Embeddings)
- **Vector Search**: ~$0.01 (OpenSearch k-NN)
- **Storage**: ~$0.01 (additional vector storage)
- **Lambda Compute**: ~$0.05 (additional processing time)

**Total RAG Cost**: ~$0.57 per 100 papers

This is a small increase compared to the existing system, but provides much more accurate redundancy detection.

## Next Steps

1. **Fine-tune similarity thresholds** based on your paper collection
2. **Monitor performance** and adjust k-NN parameters if needed
3. **Consider implementing** hybrid search (vector + text) for even better results
4. **Add support** for multi-language papers if needed

## Support

For issues with the RAG implementation:

1. Check the test results: `python test_rag_implementation.py`
2. Review Lambda logs for error messages
3. Verify environment variables and permissions
4. Test embeddings generation independently

The RAG implementation provides a significant improvement in paper deduplication accuracy while maintaining the same overall system architecture.
