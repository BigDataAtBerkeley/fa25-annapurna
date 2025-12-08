#!/bin/bash
# Simple script to kill Flask app and purge queues before redeploy

set -e

TRAINIUM_IP="${TRAINIUM_IP:-3.21.7.129}"
TRAINIUM_USER="${TRAINIUM_USER:-ec2-user}"
SSH_KEY="${1:-$SSH_KEY}"

# Try to find SSH key automatically
if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    for path in ~/.ssh/trainium-deploy-key.pem ~/.ssh/test-trn-instance.pem ~/.ssh/id_rsa ~/.ssh/trainium-key.pem; do
        if [ -f "$path" ]; then
            SSH_KEY="$path"
            break
        fi
    done
fi

if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found. Usage: $0 [path/to/key.pem]"
    exit 1
fi

echo "🔪 Step 1: Stopping Flask app on Trainium..."
ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" << 'EOF'
    # Stop systemd service first (this is what keeps restarting Flask)
    sudo systemctl stop trainium-executor 2>/dev/null || echo "Service not running or doesn't exist"
    
    # Disable auto-restart temporarily
    sudo systemctl disable trainium-executor 2>/dev/null || true
    
    # Wait for service to stop
    sleep 2
    
    # Kill any remaining Flask processes
    pkill -9 -f "python.*app.py" 2>/dev/null || true
    pkill -9 -f "flask.*run" 2>/dev/null || true
    pkill -9 -f "app.py" 2>/dev/null || true
    
    # Kill any Python processes using Neuron or trainium_exec
    ps aux | grep -E "(trainium_exec|neuron)" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    
    # Wait a moment for processes to die
    sleep 2
    
    # Restart Neuron runtime to free cores
    sudo systemctl restart neuron-rtd 2>/dev/null || echo "Neuron RTD restart skipped"
    
    # Verify Flask is dead
    if pgrep -f "app.py" > /dev/null; then
        echo "⚠️  Warning: Some Flask processes may still be running"
        pgrep -f "app.py" | xargs kill -9 2>/dev/null || true
    fi
    
    echo "✅ Flask app stopped and Neuron runtime restarted"
EOF

echo ""
echo "🗑️  Step 2: Purging trainium-execution queue..."
QUEUE_URL="https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo"

# Purge the queue
aws sqs purge-queue --queue-url "$QUEUE_URL" --region us-east-1

echo "✅ Queue purged"
echo ""
echo "📋 Next steps:"
echo "  1. Redeploy: cd deployment && ./build_code_gen_lambda.sh && ./deploy_trainium.sh $SSH_KEY"
echo "  2. Re-invoke cron: aws lambda invoke --function-name PapersCronJob --region us-east-1 response.json"

