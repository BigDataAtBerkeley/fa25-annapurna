Even though it's scheduled weekly, you can trigger it manually:

```bash
aws lambda invoke \
  --function-name PapersScraper \
  --payload '{}' \
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


