#!/bin/bash
echo "Waiting 90 seconds for rate limit to fully reset..."
sleep 90

echo "Invoking Lambda..."
aws lambda invoke \
  --function-name PapersCodeGenerator \
  --cli-binary-format raw-in-base64-out \
  --payload file://payload.json \
  --region us-east-1 \
  response.json

echo -e "\n=== Lambda Response ==="
if [ -f response.json ]; then
  cat response.json | python3 -m json.tool
else
  echo "No response file created"
fi

echo -e "\n=== Check these ==="
echo "1. Check Slack for notifications"
echo "2. Check Trainium logs: ssh -i ~/.ssh/trainium-deploy-key.pem ec2-user@3.21.7.129 'tail -50 ~/trainium-executor/logs/app.log'"
