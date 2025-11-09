#!/bin/bash
set -e

# Script to install all required packages on Trainium instance
# Run this on the Trainium instance to ensure all packages are available

echo "📦 Installing all required packages for Trainium code execution"
echo ""

# Load .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

TRAINIUM_IP="${TRAINIUM_ENDPOINT:-http://3.21.7.129:8000}"
TRAINIUM_IP=$(echo $TRAINIUM_IP | sed 's|http://||; s|https://||; s|:8000||')
SSH_KEY="${1:-$SSH_KEY}"

# Try to find SSH key automatically
if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    for path in ~/.ssh/trainium-deploy-key.pem ~/.ssh/test-trn-instance.pem ~/.ssh/id_rsa; do
        if [ -f "$path" ]; then
            SSH_KEY="$path"
            break
        fi
    done
fi

if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found. Please provide path:"
    echo "   ./deployment/install_trainium_packages.sh /path/to/key.pem"
    exit 1
fi

echo "🔧 Installing packages on Trainium instance at $TRAINIUM_IP..."
echo ""

ssh -i "$SSH_KEY" ec2-user@$TRAINIUM_IP << 'EOF'
    set -e
    
    echo "📦 Step 1: Upgrading pip..."
    pip3 install --upgrade pip --user
    
    echo ""
    echo "📦 Step 2: Installing PyTorch Neuron SDK..."
    pip3 install --user torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com
    
    echo ""
    echo "📦 Step 3: Installing HuggingFace libraries..."
    pip3 install --user datasets transformers
    
    echo ""
    echo "📦 Step 4: Installing Flask dependencies..."
    cd ~/trainium-executor
    if [ -f requirements.txt ]; then
        pip3 install --user -r requirements.txt
    fi
    
    echo ""
    echo "🔍 Step 5: Verifying installations..."
    echo ""
    
    # Check torch
    python3 -c "import torch; print(f'✓ torch {torch.__version__}')" || echo "✗ torch not found"
    
    # Check torch_xla
    python3 -c "import torch_xla.core.xla_model as xm; print('✓ torch_xla available')" || echo "✗ torch_xla not found"
    
    # Check transformers
    python3 -c "import transformers; print(f'✓ transformers {transformers.__version__}')" || echo "✗ transformers not found"
    
    # Check datasets
    python3 -c "import datasets; print(f'✓ datasets {datasets.__version__}')" || echo "✗ datasets not found"
    
    # Check numpy
    python3 -c "import numpy; print(f'✓ numpy {numpy.__version__}')" || echo "✗ numpy not found"
    
    # Check boto3
    python3 -c "import boto3; print(f'✓ boto3 {boto3.__version__}')" || echo "✗ boto3 not found"
    
    echo ""
    echo "📋 Python path information:"
    python3 -c "import sys; print('Python executable:', sys.executable); print('Python path:'); [print('  ', p) for p in sys.path]"
    
    echo ""
    echo "✅ Package installation complete!"
EOF

echo ""
echo "✅ All packages installed on Trainium instance"
echo ""
echo "⚠️  Note: If packages are installed with --user flag, they go to ~/.local/lib/python3.9/site-packages"
echo "   The Flask app should automatically find them via Python's default path."
echo ""
echo "🔄 Restarting Flask service to pick up new packages..."
ssh -i "$SSH_KEY" ec2-user@$TRAINIUM_IP "sudo systemctl restart trainium-executor && sudo systemctl status trainium-executor --no-pager"

