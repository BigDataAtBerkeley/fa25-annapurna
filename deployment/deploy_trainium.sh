#!/bin/bash
set -e

# Get script directory (absolute path) before any cd commands
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration
TRAINIUM_ENDPOINT="${TRAINIUM_ENDPOINT:-http://3.21.7.129:8000}"
TRAINIUM_IP=$(echo $TRAINIUM_ENDPOINT | sed 's|http://||; s|https://||; s|:8000||')
TRAINIUM_INSTANCE_ID="${TRAINIUM_INSTANCE_ID:-i-0f0bf0de25aa4fd57}"
TRAINIUM_USER="ec2-user"
SSH_KEY="${1:-$SSH_KEY}"

# Expand relative paths and ~
if [ -n "$SSH_KEY" ]; then
    SSH_KEY=$(eval echo "$SSH_KEY")
fi

# Try to find SSH key automatically
if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    # Try common locations
    for path in ~/.ssh/trainium-deploy-key.pem ~/.ssh/test-trn-instance.pem ~/.ssh/id_rsa ~/.ssh/trainium-key.pem; do
        if [ -f "$path" ]; then
            SSH_KEY="$path"
            break
        fi
    done
fi

# Final check - expand path one more time
if [ -n "$SSH_KEY" ]; then
    SSH_KEY=$(eval echo "$SSH_KEY")
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

# Step 0: Check and start instance if needed
echo "🔍 Step 0: Checking instance status..."
AWS_REGION="${AWS_REGION:-us-east-2}"

# Use provided instance ID or try to find it from IP
if [ -n "$TRAINIUM_INSTANCE_ID" ]; then
    INSTANCE_ID="$TRAINIUM_INSTANCE_ID"
    echo "   Using provided instance ID: $INSTANCE_ID"
else
    # Fallback: Find instance ID from IP address
    echo "   Looking up instance ID from IP $TRAINIUM_IP..."
    INSTANCE_ID=$(aws ec2 describe-instances --region "$AWS_REGION" \
        --filters "Name=ip-address,Values=$TRAINIUM_IP" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text 2>/dev/null | head -1)
    
    if [ -z "$INSTANCE_ID" ]; then
        echo "⚠️  Could not find instance ID for IP $TRAINIUM_IP"
        echo "   Attempting to continue with SSH connection test..."
        INSTANCE_ID=""
    else
        echo "   Found instance: $INSTANCE_ID"
    fi
fi

if [ -n "$INSTANCE_ID" ]; then
    
    # Check instance state (optional - skip if it causes issues)
    # Since we'll verify via SSH anyway, we can make this optional
    echo "   Checking instance state from AWS (optional)..."
    
    INSTANCE_STATE=""
    AWS_CLI_EXIT_CODE=1
    
    # Try to get state, but don't block - use a simple non-blocking approach
    # Write to a file in background, then check it
    STATE_FILE="/tmp/trainium_state_$$"
    (aws ec2 describe-instances --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null > "$STATE_FILE" &)
    
    # Give it 1 second, then check
    sleep 1
    
    if [ -f "$STATE_FILE" ] && [ -s "$STATE_FILE" ]; then
        # Got result
        INSTANCE_STATE=$(cat "$STATE_FILE" | tr -d '[:space:]')
        rm -f "$STATE_FILE"
        AWS_CLI_EXIT_CODE=0
        echo "   Instance state: $INSTANCE_STATE"
    else
        # Didn't get result in time - skip it
        rm -f "$STATE_FILE"
        echo "   ⚠️  State check skipped (will verify via SSH connection)..."
        INSTANCE_STATE=""
    fi
    
    # Check if command failed or state is invalid
    if [ $AWS_CLI_EXIT_CODE -ne 0 ] || [ -z "$INSTANCE_STATE" ] || [[ "$INSTANCE_STATE" == *"error"* ]] || [[ "$INSTANCE_STATE" == *"Error"* ]] || [[ "$INSTANCE_STATE" == *"InvalidInstanceID"* ]]; then
        echo "⚠️  Failed to get instance state (exit code: $AWS_CLI_EXIT_CODE)"
        echo "   AWS CLI output: $INSTANCE_STATE_OUTPUT"
        echo "   Attempting to continue with SSH connection test..."
        INSTANCE_STATE=""
    else
        echo "   Instance state: $INSTANCE_STATE"
    fi
    
    if [ -n "$INSTANCE_STATE" ] && [ "$INSTANCE_STATE" = "stopped" ]; then
        echo "🔄 Instance is stopped. Starting instance..."
        aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" > /dev/null
        
        if [ $? -eq 0 ]; then
            echo "   Waiting for instance to be running..."
            aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
            
            if [ $? -eq 0 ]; then
                echo "✅ Instance is now running"
                echo "   Waiting 30 seconds for SSH to be ready..."
                sleep 30
            else
                echo "❌ Failed to start instance or timeout waiting for running state"
                exit 1
            fi
        else
            echo "❌ Failed to start instance"
            exit 1
        fi
    elif [ -n "$INSTANCE_STATE" ] && [ "$INSTANCE_STATE" = "stopping" ]; then
        echo "⏳ Instance is stopping. Waiting for it to stop, then starting..."
        aws ec2 wait instance-stopped --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
        echo "🔄 Starting instance..."
        aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" > /dev/null
        aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
        echo "✅ Instance is now running"
        echo "   Waiting 30 seconds for SSH to be ready..."
        sleep 30
    elif [ -n "$INSTANCE_STATE" ] && [ "$INSTANCE_STATE" = "pending" ]; then
        echo "⏳ Instance is starting. Waiting for it to be running..."
        aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
        echo "✅ Instance is now running"
        echo "   Waiting 30 seconds for SSH to be ready..."
        sleep 30
    elif [ -n "$INSTANCE_STATE" ] && [ "$INSTANCE_STATE" = "running" ]; then
        echo "✅ Instance is already running"
    else
        echo "⚠️  Instance is in state: $INSTANCE_STATE"
        echo "   Attempting to continue..."
    fi
fi
echo ""

# Step 1: Test SSH connection
echo "🔌 Step 1: Testing SSH connection..."
MAX_SSH_RETRIES=5
SSH_RETRY_COUNT=0
SSH_SUCCESS=false

while [ $SSH_RETRY_COUNT -lt $MAX_SSH_RETRIES ]; do
    if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$TRAINIUM_USER@$TRAINIUM_IP" "echo 'Connection successful'" 2>/dev/null; then
        SSH_SUCCESS=true
        break
    fi
    
    SSH_RETRY_COUNT=$((SSH_RETRY_COUNT + 1))
    if [ $SSH_RETRY_COUNT -lt $MAX_SSH_RETRIES ]; then
        echo "   SSH connection failed, retrying in 10 seconds... (attempt $SSH_RETRY_COUNT/$MAX_SSH_RETRIES)"
        sleep 10
    fi
done

if [ "$SSH_SUCCESS" = false ]; then
    echo "❌ SSH connection failed after $MAX_SSH_RETRIES attempts!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if the instance is running:"
    if [ -n "$INSTANCE_ID" ]; then
        echo "   aws ec2 describe-instances --region $AWS_REGION --instance-ids $INSTANCE_ID"
    else
        echo "   aws ec2 describe-instances --region $AWS_REGION --filters 'Name=ip-address,Values=$TRAINIUM_IP'"
    fi
    echo ""
    echo "2. Verify security group allows SSH (port 22) from your IP"
    echo ""
    echo "3. Check SSH key permissions:"
    echo "   chmod 400 $SSH_KEY"
    echo ""
    echo "4. Try with absolute path:"
    echo "   ./deployment/deploy_trainium.sh ~/.ssh/trainium-deploy-key.pem"
    echo ""
    exit 1
fi
echo "✅ SSH connection OK"
echo ""

SSH_KEY=$(cd "$(dirname "$SSH_KEY")" && pwd)/$(basename "$SSH_KEY")

# Step 2: Upload files
echo "📦 Step 2: Uploading app files..."
cd "$SCRIPT_DIR/../trn_execute" || exit 1

# Check files exist
for file in app.py error_db.py s3_code_storage.py requirements.txt opensearch_client.py sagemaker_metrics.py dataset_loader.py slack_notifier.py; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found in trn_execute"
        exit 1
    fi
done

echo "  Uploading: app.py, error_db.py, s3_code_storage.py, opensearch_client.py, requirements.txt, sagemaker_metrics.py, dataset_loader.py, slack_notifier.py"
echo " Little easter egg for you :) - Tarun says hi!"
scp -i "$SSH_KEY" -o ConnectTimeout=30 -v app.py error_db.py s3_code_storage.py opensearch_client.py requirements.txt sagemaker_metrics.py dataset_loader.py slack_notifier.py "$TRAINIUM_USER@$TRAINIUM_IP:~/" 2>&1 | grep -E "(Sending|100%)" || true
echo "✅ Files uploaded"
echo ""

# Step 3: Install dependencies and start Flask
echo "⚙️  Step 3: Installing dependencies on Trainium..."
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" << 'EOF'
    # Update system
    echo "Installing system packages..."
    sudo yum update -y
    sudo yum install -y python3 python3-pip
    
    # Install PyTorch Neuron SDK (includes torch_xla for Trainium)
    echo "Installing PyTorch Neuron SDK..."
    pip3 install --upgrade pip
    pip3 install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com
    
    # Verify torch_xla is available (it comes with torch-neuronx)
    echo "Verifying torch_xla installation..."
    python3 -c "import torch_xla.core.xla_model as xm; print('✓ torch_xla available')" || \
    echo "⚠️  torch_xla import failed - check Neuron SDK installation"
    
    # Install Flask dependencies (system Python for Flask app)
    echo "Installing Flask dependencies..."
    pip3 install -r requirements.txt
    
    # Create working directory
    mkdir -p ~/trainium-executor
    mv ~/app.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/requirements.txt ~/trainium-executor/ 2>/dev/null || true
    mv ~/sagemaker_metrics.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/dataset_loader.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/error_db.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/s3_code_storage.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/slack_notifier.py ~/trainium-executor/ 2>/dev/null || true
    mv ~/opensearch_client.py ~/trainium-executor/ 2>/dev/null || true
    
    echo "✅ System installation complete"
EOF

# Step 3b: Install packages in Neuron venv (for generated code execution)
echo ""
echo "📦 Step 3b: Installing packages in Neuron venv for generated code..."
INSTALL_SCRIPT="$SCRIPT_DIR/install_all_packages_trainium.sh"

if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "❌ Error: install_all_packages_trainium.sh not found at $INSTALL_SCRIPT"
    exit 1
fi

scp -i "$SSH_KEY" "$INSTALL_SCRIPT" "$TRAINIUM_USER@$TRAINIUM_IP:~/"
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" << 'EOF'
    chmod +x ~/install_all_packages_trainium.sh
    # Run non-interactively (defaults to yes for tokenizers)
    SKIP_TOKENIZER_DOWNLOAD=false ~/install_all_packages_trainium.sh || {
        echo "⚠️  Package installation had issues, but continuing..."
    }
    rm ~/install_all_packages_trainium.sh
EOF

echo ""
echo "🔧 Step 4: Setting up systemd service..."
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" << 'EOF'
    # Create logs directory first (before systemd service)
    mkdir -p ~/trainium-executor/logs
    chmod 755 ~/trainium-executor/logs
    
    # Create systemd service file
    sudo tee /etc/systemd/system/trainium-executor.service > /dev/null << 'SERVICE'
[Unit]
Description=Trainium Code Executor Flask API
After=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/trainium-executor
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}"
ExecStart=/usr/bin/python3 /home/ec2-user/trainium-executor/app.py
Restart=always
RestartSec=10
# Log to both journal and file
StandardOutput=journal+append:/home/ec2-user/trainium-executor/logs/app.log
StandardError=journal+append:/home/ec2-user/trainium-executor/logs/error.log

[Install]
WantedBy=multi-user.target
SERVICE

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable trainium-executor
    
    # Stop service if it's running (to restart cleanly)
    sudo systemctl stop trainium-executor 2>/dev/null || true
    
    # Start service
    if sudo systemctl start trainium-executor; then
        echo "✅ Service started successfully"
        sleep 2
        # Check if it's actually running
        if sudo systemctl is-active --quiet trainium-executor; then
            echo "✅ Service is active"
        else
            echo "⚠️  Service started but may have failed - check logs:"
            echo "   sudo journalctl -u trainium-executor -n 50 --no-pager"
            echo "   tail -50 ~/trainium-executor/logs/error.log"
        fi
    else
        echo "❌ Failed to start service"
        echo "Check logs:"
        echo "   sudo journalctl -u trainium-executor -n 50 --no-pager"
        exit 1
    fi
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Service Status:"
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "sudo systemctl status trainium-executor --no-pager -l"
echo ""

# Wait a bit for service to fully start
echo "⏳ Waiting for service to start..."
sleep 5

# Check service status
echo "📋 Checking service status..."
if ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "sudo systemctl is-active --quiet trainium-executor"; then
    echo "✅ Service is running"
else
    echo "❌ Service is not running!"
    echo ""
    echo "📋 Recent service logs:"
    ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "sudo journalctl -u trainium-executor -n 30 --no-pager -l" || true
    echo ""
    echo "📋 Error log file:"
    ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "tail -50 ~/trainium-executor/logs/error.log 2>/dev/null || echo 'Error log file not found'" || true
    echo ""
    echo "📋 App log file:"
    ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "tail -50 ~/trainium-executor/logs/app.log 2>/dev/null || echo 'App log file not found'" || true
    echo ""
    exit 1
fi

echo ""
echo "🧪 Testing health endpoint..."
sleep 2
if curl -s --max-time 10 "http://$TRAINIUM_IP:8000/health" | python3 -m json.tool; then
    echo "✅ Health check passed!"
else
    echo "⚠️  Health check failed"
    echo "   This might be a security group issue (port 8000 not open)"
    echo "   Or the service might still be starting up"
    echo ""
    echo "   Check service logs:"
    echo "   ssh -i $SSH_KEY $TRAINIUM_USER@$TRAINIUM_IP 'sudo journalctl -u trainium-executor -f'"
fi
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

