#!/bin/bash
#
# Setup script for Trainium Executor Service on trn1.2xlarge instance.
#
# Run this script after launching a trn1 instance with Deep Learning AMI.
#

set -e

echo "========================================="
echo "Trainium Executor Setup"
echo "========================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Python and pip
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3-pip python3-venv python3-dev

# Install system dependencies
sudo apt install -y build-essential git curl wget

# Configure Neuron repositories (if not already configured)
if [ ! -f /etc/apt/sources.list.d/neuron.list ]; then
    echo "⚙️  Configuring Neuron repositories..."
    . /etc/os-release
    sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
    sudo apt update
fi

# Install Neuron drivers and tools
echo "🧠 Installing AWS Neuron SDK..."
sudo apt install -y aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools

# Create application directory
APP_DIR="/opt/trainium-executor"
echo "📁 Creating application directory: $APP_DIR"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Copy application files
echo "📋 Copying application files..."
cp app.py $APP_DIR/
cp requirements.txt $APP_DIR/

# Create Python virtual environment
echo "🔧 Creating Python virtual environment..."
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with Neuron support
echo "🔥 Installing PyTorch with Neuron support..."
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com
pip install torch==2.1.0

# Create working directory for jobs
WORKING_DIR="/var/lib/trainium-jobs"
echo "📁 Creating working directory: $WORKING_DIR..."
sudo mkdir -p $WORKING_DIR
sudo chown $USER:$USER $WORKING_DIR

# Create log directory
LOG_DIR="/var/log"
sudo touch $LOG_DIR/trainium-executor.log
sudo chown $USER:$USER $LOG_DIR/trainium-executor.log

# Create systemd service
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/trainium-executor.service > /dev/null <<EOF
[Unit]
Description=Trainium Executor Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="WORKING_DIR=$WORKING_DIR"
Environment="MAX_EXECUTION_TIME=600"
Environment="NEURON_RT_LOG_LEVEL=ERROR"
ExecStart=$APP_DIR/venv/bin/python3 $APP_DIR/app.py
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/trainium-executor.log
StandardError=append:$LOG_DIR/trainium-executor.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
echo "🔄 Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable trainium-executor

# Start service
echo "🚀 Starting service..."
sudo systemctl start trainium-executor

# Check status
echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "Service status:"
sudo systemctl status trainium-executor --no-pager

echo ""
echo "📝 Useful commands:"
echo "  - View logs: sudo journalctl -u trainium-executor -f"
echo "  - Restart service: sudo systemctl restart trainium-executor"
echo "  - Stop service: sudo systemctl stop trainium-executor"
echo "  - Check status: sudo systemctl status trainium-executor"
echo "  - Test health: curl http://localhost:8000/health"
echo ""
echo "🌐 The service is listening on port 8000"
echo "   Make sure your security group allows inbound TCP 8000 from Lambda VPC"
echo ""

