#!/bin/bash
#
# Build and package the Cron Lambda function
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAMBDA_DIR="$PROJECT_ROOT/cron_lambda"
BUILD_DIR="$LAMBDA_DIR/build"
PACKAGE_DIR="$BUILD_DIR/package"

echo "========================================="
echo "Building Cron Lambda"
echo "========================================="
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf "$BUILD_DIR"
mkdir -p "$PACKAGE_DIR"

# Copy Lambda function
echo "📦 Copying Lambda function..."
cp "$LAMBDA_DIR/lambda_function.py" "$PACKAGE_DIR/"

# Install dependencies
echo "📥 Installing dependencies..."
cd "$PACKAGE_DIR"
pip install -r "$LAMBDA_DIR/requirements.txt" -t . --quiet

# Remove unnecessary files to reduce package size
echo "🧹 Removing unnecessary files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Create deployment package
echo "📦 Creating deployment package..."
cd "$PACKAGE_DIR"
zip -r "$BUILD_DIR/cron_lambda.zip" . -q

echo ""
echo "✅ Build complete!"
echo "📦 Package: $BUILD_DIR/cron_lambda.zip"
echo ""
echo "📋 Next steps:"
echo "  1. Upload to Lambda:"
echo "     aws lambda update-function-code \\"
echo "       --function-name PapersCronJob \\"
echo "       --zip-file fileb://$BUILD_DIR/cron_lambda.zip \\"
echo "       --region us-east-1"
echo ""
echo "  2. Or create new Lambda function:"
echo "     aws lambda create-function \\"
echo "       --function-name PapersCronJob \\"
echo "       --runtime python3.11 \\"
echo "       --role <LAMBDA_ROLE_ARN> \\"
echo "       --handler lambda_function.lambda_handler \\"
echo "       --zip-file fileb://$BUILD_DIR/cron_lambda.zip \\"
echo "       --timeout 300 \\"
echo "       --memory-size 512 \\"
echo "       --region us-east-1"
echo ""

