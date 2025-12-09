#!/bin/bash
# Check if system is ready to re-invoke cron job

set -e

TRAINIUM_IP="${TRAINIUM_IP:-3.21.7.129}"
TRAINIUM_USER="${TRAINIUM_USER:-ec2-user}"
SSH_KEY="${1:-$SSH_KEY}"
QUEUE_URL="https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo"

# Try to find SSH key automatically
if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    for path in ~/.ssh/trainium-deploy-key.pem ~/.ssh/test-trn-instance.pem ~/.ssh/id_rsa ~/.ssh/trainium-key.pem; do
        if [ -f "$path" ]; then
            SSH_KEY="$path"
            break
        fi
    done
fi

echo "🔍 Checking if system is ready for cron job..."
echo ""

# Check 1: Flask app status
echo "1. Checking Flask app status..."
if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
    FLASK_RUNNING=$(ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "pgrep -f 'python.*app.py' || echo ''" 2>/dev/null || echo "")
    if [ -z "$FLASK_RUNNING" ]; then
        echo "   ✅ Flask app is stopped"
    else
        echo "   ⚠️  Flask app is still running (PID: $FLASK_RUNNING)"
        echo "      Run: ./kill_and_purge.sh to stop it"
    fi
else
    echo "   ⚠️  Cannot check Flask app (SSH key not found)"
fi

# Check 2: Queue status
echo ""
echo "2. Checking trainium-execution queue..."
QUEUE_ATTRS=$(aws sqs get-queue-attributes \
    --queue-url "$QUEUE_URL" \
    --attribute-names ApproximateNumberOfMessages ApproximateNumberOfMessagesNotVisible \
    --region us-east-1 \
    --output json 2>/dev/null || echo '{}')

if [ "$QUEUE_ATTRS" != "{}" ]; then
    VISIBLE=$(echo "$QUEUE_ATTRS" | jq -r '.Attributes.ApproximateNumberOfMessages // "0"')
    IN_FLIGHT=$(echo "$QUEUE_ATTRS" | jq -r '.Attributes.ApproximateNumberOfMessagesNotVisible // "0"')
    TOTAL=$((VISIBLE + IN_FLIGHT))
    
    if [ "$TOTAL" -eq 0 ]; then
        echo "   ✅ Queue is empty"
    else
        echo "   ⚠️  Queue has $TOTAL messages ($VISIBLE visible, $IN_FLIGHT in flight)"
        echo "      Run: aws sqs purge-queue --queue-url $QUEUE_URL --region us-east-1"
    fi
else
    echo "   ⚠️  Could not check queue status"
fi

# Check 3: Neuron processes
echo ""
echo "3. Checking Neuron processes..."
if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
    NEURON_PROCS=$(ssh -i "$SSH_KEY" "$TRAINIUM_USER@$TRAINIUM_IP" "ps aux | grep -E '(trainium_exec|neuron)' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    if [ "$NEURON_PROCS" -eq 0 ]; then
        echo "   ✅ No Neuron processes running"
    else
        echo "   ⚠️  Found $NEURON_PROCS Neuron-related processes"
        echo "      Run: ./kill_and_purge.sh to clean them up"
    fi
else
    echo "   ⚠️  Cannot check Neuron processes (SSH key not found)"
fi

# Check 4: Health endpoint (if Flask is running)
echo ""
echo "4. Checking Trainium endpoint..."
HEALTH=$(curl -s -m 2 "http://$TRAINIUM_IP:8000/health" 2>/dev/null || echo "")
if [ -n "$HEALTH" ]; then
    echo "   ⚠️  Flask app is responding (should be stopped)"
else
    echo "   ✅ Flask app is not responding (stopped)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Summary
ALL_CLEAR=true

# If Flask endpoint is not responding, Flask is effectively stopped (most important check)
if [ -n "$HEALTH" ]; then
    ALL_CLEAR=false
    echo "   ❌ Flask endpoint is responding (should be stopped)"
else
    echo "   ✅ Flask endpoint is not responding (stopped - this is what matters)"
fi

if [ "$TOTAL" -gt 0 ] 2>/dev/null; then
    ALL_CLEAR=false
fi

# Neuron processes are OK if Flask is stopped (they'll be cleaned up on next start)
if [ "$NEURON_PROCS" -gt 0 ] 2>/dev/null && [ -z "$HEALTH" ]; then
    echo "   ℹ️  Some Neuron processes found, but Flask is stopped (will be cleaned on restart)"
fi

if [ "$ALL_CLEAR" = true ]; then
    echo ""
    echo "✅ System is ready! You can safely re-invoke the cron job:"
    echo ""
    echo "   aws lambda invoke --function-name PapersCronJob --region us-east-1 response.json"
else
    echo ""
    echo "⚠️  System is NOT ready. Please run:"
    echo ""
    echo "   ./kill_and_purge.sh [path/to/ssh-key.pem]"
    echo ""
    echo "Then check again with: ./check_ready.sh"
fi

