Even though it's scheduled weekly, you can trigger it manually:

```bash

# Scrape ICLR papers
aws lambda invoke --function-name PaperScraper_ICLR --payload '{"MAX_PAPERS": "5"}' --cli-binary-format raw-in-base64-out scraper_output.json


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

# Scrape both ICLR and ICML papers
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{"CONFERENCE": "BOTH"}' \
  --cli-binary-format raw-in-base64-out \
  scraper_output.json && cat scraper_output.json



## CHECK LOGS

1. ICLR Scraper logs:
aws logs tail /aws/lambda/PaperScraper_ICLR --since 5m --follow

**TO CHECK OTHER SCRAPER LOGS JUST REPLACE "ICLR" WITH "ICML", "ARXIV", etc**

2. Judge logs
aws logs tail /aws/lambda/PapersJudge --since 15m --follow


## REBUILDING JUDGE LAMBDA
bash build_judge.sh



## Check SQS Queue (if not empty, the judge lambda didn't process the documents in the queue)
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/478852001205/researchQueue.fifo \
  --attribute-names ApproximateNumberOfMessages


## Check OpenSearch documents and health
To verify indexed papers:
python check_opensearch.py

## Clear OpenSearch (DO NOT DO THIS BEFORE SPECIFICALLY ASKING DAN)
python clear_opensearch.py

AWS_REGION=us-east-1
OPENSEARCH_ENDPOINT=your-opensearch-endpoint
OPENSEARCH_INDEX=research-papers
CLAUDE_API_KEY=your-claude-key

## Environment Variables

### Scraper Lambda
- `CONFERENCE`: Conference to scrape ("ICLR", "ICML", or "BOTH") - default: "ICLR"
- `CONFERENCE_YEAR`: Year to scrape (e.g., "2025") - default: "2025"
- `MAX_PAPERS`: Maximum papers to process - default: "3"
- `BUCKET_NAME`: S3 bucket for PDF storage
- `QUEUE_URL`: SQS queue URL for paper metadata

### Judge Lambda  
- `OPENSEARCH_ENDPOINT`: OpenSearch cluster endpoint
- `OPENSEARCH_INDEX`: Index name for storing papers - default: "research-papers"


