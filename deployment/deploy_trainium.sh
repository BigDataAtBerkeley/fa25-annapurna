#!/bin/bash
set -e

# Load .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration
TRAINIUM_IP="${TRAINIUM_ENDPOINT:-http://3.134.87.226:8000}"
TRAINIUM_IP=$(echo $TRAINIUM_IP | sed 's|http://||; s|https://||; s|:8000||')
TRAINIUM_USER="ec2-user"
SSH_KEY="${1:-$SSH_KEY}"

# Try to find SSH key automatically
if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    # Try common locations
    for path in ~/.ssh/test-trn-instance.pem ~/.ssh/id_rsa ~/.ssh/trainium-key.pem; do
        if [ -f "$path" ]; then
            SSH_KEY="$path"
            break
        fi
    done
fi

if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found"
    echo ""
    echo "The key pair 'test-trn-instance' exists in AWS, but we need the private .pem file locally."
    echo ""
    echo "Options:"
    echo "1. If you have the key file, provide the path:"
    echo "   ./deployment/deploy_trainium.sh /path/to/test-trn-instance.pem"
    echo ""
    echo "2. Or set SSH_KEY in your .env file:"
    echo "   SSH_KEY=/path/to/test-trn-instance.pem"
    echo ""
    echo "3. If you don't have the key, you'll need to:"
    echo "   - Create a new key pair in AWS Console"
    echo "   - Download the .pem file"
    echo "   - Or use AWS Systems Manager (requires IAM role setup)"
    echo ""
    exit 1
fi

echo "🚀 Deploying Flask app to Trainium instance at $TRAINIUM_IP"
echo ""

# Step 1: Upload files
echo "📦 Step 1: Uploading app files..."
cd "$(dirname "$0")/../trainium_executor" || exit 1
scp -i "$SSH_KEY" app.py requirements.txt sagemaker_metrics.py dataset_loader.py "$TRAINIUM_USER@$TRAINIUM_IP:~/"
echo "✅ Files uploaded"
echo ""

# Step 2: Install dependencies and start Flask
echo "⚙️  Step 2: Installing dependencies on Trainium..."
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" << 'EOF'
    # Update system
    echo "Installing system packages..."
    sudo yum update -y
    sudo yum install -y python3 python3-pip
    
    # Install PyTorch Neuron SDK
    echo "Installing PyTorch Neuron..."
    pip3 install --upgrade pip
    pip3 install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com
    
    # Install torchvision (needed for dataset loaders)
    echo "Installing torchvision..."
    pip3 install torchvision
    
    # Install Flask dependencies
    echo "Installing Flask dependencies..."
    pip3 install -r requirements.txt
    
    # Create working directory
    mkdir -p ~/trainium-executor
    mv ~/app.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/requirements.txt ~/trainium-executor/ 2>/dev/null || true
    mv ~/sagemaker_metrics.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/dataset_loader.py ~/trainium-executor/ 2>/dev/null || true
    
    echo "✅ Installation complete"
EOF

echo ""
echo "🔧 Step 3: Setting up systemd service..."
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" << 'EOF'
    # Create systemd service file
    sudo tee /etc/systemd/system/trainium-executor.service > /dev/null << 'SERVICE'
[Unit]
Description=Trainium Code Executor Flask API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/trainium-executor
ExecStart=/usr/bin/python3 /home/ec2-user/trainium-executor/app.py
Restart=always
RestartSec=10
StandardOutput=append:/home/ec2-user/trainium-executor/logs/app.log
StandardError=append:/home/ec2-user/trainium-executor/logs/error.log

[Install]
WantedBy=multi-user.target
SERVICE

    # Create logs directory
    mkdir -p ~/trainium-executor/logs
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable trainium-executor
    sudo systemctl start trainium-executor
    
    echo "✅ Service started"
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Service Status:"
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "sudo systemctl status trainium-executor --no-pager"
echo ""
echo "🧪 Testing health endpoint..."
sleep 3
curl -s "http://$TRAINIUM_IP:8000/health" | python3 -m json.tool || echo "⚠️  Health check failed - may need to configure security group"
echo ""
echo "============================================================"
echo "📋 Next Steps:"
echo "============================================================"
echo "1. Security group should already allow port 8000"
echo ""
echo "2. Add inbound rule:"
echo "   aws ec2 authorize-security-group-ingress --region us-east-2 \\"
echo "     --group-id <SECURITY_GROUP_ID> \\"
echo "     --protocol tcp --port 8000 --cidr 0.0.0.0/0"
echo ""
echo "3. Test again:"
echo "   curl http://$TRAINIUM_IP:8000/health"
echo ""
echo "4. View logs:"
echo "   ssh -i $SSH_KEY $TRAINIUM_USER@$TRAINIUM_IP 'tail -f ~/trainium-executor/logs/app.log'"

