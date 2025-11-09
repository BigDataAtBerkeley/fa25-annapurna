#!/bin/bash
# Run this script ON the Trainium instance (us-east-2)
# It will download the Flask app from S3 and start it

set -e

echo "🚀 Setting up Trainium Code Executor"
echo ""

# Update system
echo "📦 Installing system packages..."
sudo yum update -y
sudo yum install -y python3 python3-pip git

# Install PyTorch Neuron SDK (includes torch_xla for Trainium)
echo "🧠 Installing PyTorch Neuron SDK..."
pip3 install --upgrade pip
pip3 install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com

# torch_xla is included with torch-neuronx, no separate installation needed
echo "✓ torch_xla is included with torch-neuronx"

# Create working directory
echo "📁 Creating working directory..."
mkdir -p ~/trainium-executor/logs
cd ~/trainium-executor

# Download files from S3
echo "⬇️  Downloading app files from S3..."
aws s3 cp s3://papers-test-outputs/setup/app.py ./
aws s3 cp s3://papers-test-outputs/setup/requirements.txt ./

# Install Flask dependencies
echo "📦 Installing Flask dependencies..."
pip3 install -r requirements.txt

# Create systemd service
echo "⚙️  Creating systemd service..."
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
Environment="PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
SERVICE

# Enable and start service
echo "🔥 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable trainium-executor
sudo systemctl start trainium-executor

sleep 3

# Check status
echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Service Status:"
sudo systemctl status trainium-executor --no-pager
echo ""
echo "🧪 Testing health endpoint..."
curl -s http://localhost:8000/health | python3 -m json.tool || echo "⚠️  Health check failed"
echo ""
echo "============================================================"
echo "📋 Useful Commands:"
echo "============================================================"
echo "Check status:  sudo systemctl status trainium-executor"
echo "View logs:     tail -f ~/trainium-executor/logs/app.log"
echo "Restart:       sudo systemctl restart trainium-executor"
echo "Stop:          sudo systemctl stop trainium-executor"
echo ""
echo "🌐 Configure security group to allow port 8000 from Lambda"

