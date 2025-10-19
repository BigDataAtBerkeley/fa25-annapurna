Even though it's scheduled weekly, you can trigger it manually:

```bash
# Scrape ICLR papers (default)
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json

# Scrape ICML papers
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "ICML"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json

# Scrape ICML papers filtered by topic (Large Language Models)
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "ICML", "topic_filter": "Deep Learning->Large Language Models"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json

# Scrape from custom ICML URL
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "ICML", "custom_url": "https://icml.cc/virtual/2025/papers.html?filter=topic&search=Deep+Learning-%3ELarge+Language+Models"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json

# Scrape ACL 2025 papers (focused on LLMs)
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "ACL", "keyword_filter": "LLM"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json

# Scrape ACL papers with custom keyword filter
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "ACL", "keyword_filter": "transformer"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json

# Scrape both ICLR and ICML papers
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "BOTH"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json



CHECK LOGS

1. Scraper logs:
aws logs tail /aws/lambda/PapersScraper --since 15m --follow

2. Judge logs
aws logs tail /aws/lambda/PapersJudge --since 15m --follow



Check SQS Queue
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo \
  --attribute-names ApproximateNumberOfMessages


Check OpenSearch
To verify indexed papers:
python check_opensearch.py


AWS_REGION=us-east-1
OPENSEARCH_ENDPOINT=your-opensearch-endpoint
OPENSEARCH_INDEX=research-papers
CLAUDE_API_KEY=your-claude-key

## Environment Variables

### Scraper Lambda
- `CONFERENCE`: Conference to scrape ("ICLR", "ICML", "ACL", or "BOTH") - default: "ICLR"
- `CONFERENCE_YEAR`: Year to scrape (e.g., "2025") - default: "2025"
- `MAX_PAPERS`: Maximum papers to process - default: "3"
- `BUCKET_NAME`: S3 bucket for PDF storage
- `QUEUE_URL`: SQS queue URL for paper metadata

### Judge Lambda  
- `OPENSEARCH_ENDPOINT`: OpenSearch cluster endpoint
- `OPENSEARCH_INDEX`: Index name for storing papers - default: "research-papers"


