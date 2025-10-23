#!/bin/bash
set -e

# Configuration
TRAINIUM_IP="3.19.105.192"
TRAINIUM_USER="ec2-user"
SSH_KEY="${1:-$HOME/.ssh/your-key.pem}"  # Pass SSH key as first argument

if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "Usage: ./deploy_to_trainium.sh /path/to/your-key.pem"
    exit 1
fi

echo "🚀 Deploying Flask app to Trainium instance at $TRAINIUM_IP"
echo ""

# Step 1: Upload files
echo "📦 Step 1: Uploading app files..."
scp -i "$SSH_KEY" app.py requirements.txt "$TRAINIUM_USER@$TRAINIUM_IP:~/"
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
    
    # Install Flask dependencies
    echo "Installing Flask dependencies..."
    pip3 install -r requirements.txt
    
    # Create working directory
    mkdir -p ~/trainium-executor
    mv ~/app.py ~/trainium-executor/
    mv ~/requirements.txt ~/trainium-executor/
    
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
echo "1. Configure security group to allow port 8000:"
echo "   aws ec2 describe-instances --region us-east-2 --instance-ids i-07f2dceab59b6b44e --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text"
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

