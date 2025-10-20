#!/bin/bash

rm -f PapersJudge.zip
cd judge_lambda
zip -r ../PapersJudge.zip . -x "*.log" "logs/*" "__pycache__/*" "*.pyc" "*.zip"
cd ..
aws lambda update-function-code \
  --function-name PapersJudge \
  --zip-file fileb://PapersJudge.zip

echo "Judge Lambda updated successfully"