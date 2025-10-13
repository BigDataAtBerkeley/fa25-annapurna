#!/bin/bash

echo "Deploying all Lambda functions..."

# Deploy Judge Lambda
echo "1. Deploying Judge Lambda..."
bash build_judge.sh

# Deploy all Scraper Lambdas
echo "2. Deploying ICLR Scraper..."
bash build_scraper.sh PaperScraper_ICLR

echo "3. Deploying ICML Scraper..."
bash build_scraper.sh PaperScraper_ICML

echo "4. Deploying ArXiv Scraper..."
bash build_scraper.sh PaperScraper_arxiv

echo "4. Deploying ArXiv Scraper..."
bash build_scraper.sh PaperScraper_NEURIPS

echo "4. Deploying ArXiv Scraper..."
bash build_scraper.sh PaperScraper_MLSYS

echo "All Lambda functions deployed successfully!"