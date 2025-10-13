#!/bin/bash

echo "Updating Lambda function handlers..."

# Scraper functions use lambda_handler.lambda_handler
for func in PaperScraper_ICML PaperScraper_NEURIPS PaperScraper_ICLR PaperScraper_MLSYS PaperScraper_arxiv; do
    echo "Updating $func handler..."
    aws lambda update-function-configuration \
        --function-name $func \
        --handler lambda_handler.lambda_handler
done

# Judge function uses lambda_function.lambda_handler
echo "Updating PapersJudge handler..."
aws lambda update-function-configuration \
    --function-name PapersJudge \
    --handler lambda_function.lambda_handler

echo "All handlers updated successfully!"