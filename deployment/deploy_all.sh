#!/bin/bash
set -e

# Deploy all Lambda functions
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Deploying all Lambda functions..."
echo "=========================================="
echo ""

# Deploy Judge Lambda
echo "[1/6] Deploying Judge Lambda..."
./build_judge.sh
echo ""

# Deploy all Scraper Lambdas
echo "[2/6] Deploying ICLR Scraper..."
./build_scraper.sh PaperScraper_ICLR
echo ""

echo "[3/6] Deploying ICML Scraper..."
./build_scraper.sh PaperScraper_ICML
echo ""

echo "[4/6] Deploying ArXiv Scraper..."
./build_scraper.sh PaperScraper_arxiv
echo ""

echo "[5/6] Deploying NEURIPS Scraper..."
./build_scraper.sh PaperScraper_NEURIPS
echo ""

echo "[6/6] Deploying MLSYS Scraper..."
./build_scraper.sh PaperScraper_MLSYS
echo ""

echo "=========================================="
echo "✅ All Lambda functions deployed successfully!"
echo "=========================================="