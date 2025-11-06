#!/bin/bash
# Tail Trainium execution logs in real-time
# Usage: ./tail_trainium_logs.sh [paper_id]

set -e

# Load environment variables (handle values with spaces/special chars)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Get Trainium IP from endpoint or use provided
if [ -z "$TRAINIUM_ENDPOINT" ]; then
    echo "❌ TRAINIUM_ENDPOINT not set in .env"
    echo "Usage: ./tail_trainium_logs.sh [paper_id]"
    echo "Or set TRAINIUM_ENDPOINT in .env"
    exit 1
fi

# Extract IP from endpoint
TRAINIUM_IP=$(echo $TRAINIUM_ENDPOINT | sed 's|http://||; s|https://||; s|:8000||')

# Find SSH key automatically
find_ssh_key() {
    # If SSH_KEY is explicitly set and exists, use it
    if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
        echo "$SSH_KEY"
        return 0
    fi
    
    # Try to get key name from instance if TRAINIUM_INSTANCE_ID is set
    if [ -n "$TRAINIUM_INSTANCE_ID" ]; then
        # Account enabled for trn1 in us-east-2 and us-west-2
        TRAINIUM_REGION="${TRAINIUM_REGION:-us-east-2}"
        KEY_NAME=$(aws ec2 describe-instances \
            --region "$TRAINIUM_REGION" \
            --instance-ids "$TRAINIUM_INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].KeyName' \
            --output text 2>/dev/null)
        
        if [ -n "$KEY_NAME" ] && [ "$KEY_NAME" != "None" ]; then
            # Try common locations with the key name
            POSSIBLE_LOCATIONS=(
                "$HOME/.ssh/${KEY_NAME}.pem"
                "$HOME/.ssh/${KEY_NAME}"
                "$HOME/Downloads/${KEY_NAME}.pem"
                "$HOME/Desktop/${KEY_NAME}.pem"
                "./${KEY_NAME}.pem"
            )
            
            for location in "${POSSIBLE_LOCATIONS[@]}"; do
                if [ -f "$location" ] && grep -q "BEGIN.*PRIVATE KEY" "$location" 2>/dev/null; then
                    echo "$location"
                    return 0
                fi
            done
        fi
    fi
    
    # Fallback to common default locations
    DEFAULT_LOCATIONS=(
        "$HOME/.ssh/trainium-key.pem"
        "$HOME/.ssh/id_rsa"
        "$HOME/.ssh/id_ed25519"
    )
    
    for location in "${DEFAULT_LOCATIONS[@]}"; do
        if [ -f "$location" ] && grep -q "BEGIN.*PRIVATE KEY" "$location" 2>/dev/null; then
            echo "$location"
            return 0
        fi
    done
    
    return 1
}

# Find SSH key
SSH_KEY=$(find_ssh_key)

if [ -z "$SSH_KEY" ] || [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found"
    echo ""
    if [ -n "$TRAINIUM_INSTANCE_ID" ]; then
        # Account enabled for trn1 in us-east-2 and us-west-2
        TRAINIUM_REGION="${TRAINIUM_REGION:-us-east-2}"
        KEY_NAME=$(aws ec2 describe-instances \
            --region "$TRAINIUM_REGION" \
            --instance-ids "$TRAINIUM_INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].KeyName' \
            --output text 2>/dev/null)
        if [ -n "$KEY_NAME" ] && [ "$KEY_NAME" != "None" ]; then
            echo "   Instance uses key: $KEY_NAME"
            echo "   Look for: ${KEY_NAME}.pem in ~/.ssh, ~/Downloads, or ~/Desktop"
        fi
    fi
    echo ""
    echo "   Set SSH_KEY environment variable:"
    echo "   export SSH_KEY=\"/path/to/your-key.pem\""
    echo ""
    echo "   Or place key at one of:"
    echo "   - $HOME/.ssh/trainium-key.pem"
    echo "   - $HOME/.ssh/id_rsa"
    exit 1
fi

PAPER_ID="$1"

echo "🔍 Tailing Trainium logs..."
echo "   Instance: $TRAINIUM_IP"
echo "   Log file: ~/trainium-executor/logs/trainium-executor.log"
if [ -n "$PAPER_ID" ]; then
    echo "   Filtering for paper: $PAPER_ID"
    echo ""
    ssh -i "$SSH_KEY" ec2-user@$TRAINIUM_IP "tail -f ~/trainium-executor/logs/trainium-executor.log | grep --line-buffered '$PAPER_ID'"
else
    echo ""
    ssh -i "$SSH_KEY" ec2-user@$TRAINIUM_IP 'tail -f ~/trainium-executor/logs/trainium-executor.log'
fi

