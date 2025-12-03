#!/bin/bash
aws lambda update-function-configuration \
  --function-name PapersCodeGenerator \
  --region us-east-1 \
  --environment "Variables={
    OPENSEARCH_ENDPOINT=search-research-papers-uv3fxq76j5bkxq3bgp3nyfdtnm.us-east-1.es.amazonaws.com,
    OPENSEARCH_INDEX=research-papers-v2,
    CODE_TEST_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/478852001205/code-testing.fifo,
    CODE_BUCKET=papers-code-artifacts,
    BEDROCK_MODEL_ID=arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0,
    FLASK_EXECUTE_ENDPOINT=http://3.21.7.129:8000/execute,
    SLACK_BOT_TOKEN=xoxb-552112250854-10031003801584-OFAzmiCTvAsECqlzIKmy9Ck1,
    SLACK_CHANNEL=apl-research-papers
  }"
