#!/bin/bash
#
# Check when the next cron job will run
#

AWS_REGION="${AWS_REGION:-us-east-1}"
RULE_NAME="papers-cron-job-1hour"
LOG_GROUP="/aws/lambda/PapersCronJob"

echo "📅 Cron Job Schedule Check"
echo "=========================="
echo ""

# Get rule information
RULE_INFO=$(aws events describe-rule --name "$RULE_NAME" --region "$AWS_REGION" --output json 2>/dev/null)

if [ -z "$RULE_INFO" ]; then
    echo "❌ Rule '$RULE_NAME' not found"
    exit 1
fi

SCHEDULE=$(echo "$RULE_INFO" | jq -r '.ScheduleExpression')
STATE=$(echo "$RULE_INFO" | jq -r '.State')

echo "Rule: $RULE_NAME"
echo "Schedule: $SCHEDULE"
echo "State: $STATE"
echo ""

# Get last execution
LAST_EXEC_JSON=$(aws logs filter-log-events \
    --log-group-name "$LOG_GROUP" \
    --region "$AWS_REGION" \
    --start-time $(($(date +%s) - 86400))000 \
    --filter-pattern "Starting cron job execution" \
    --query 'events[-1].timestamp' \
    --output json 2>/dev/null)

# Parse timestamp (handle JSON format)
LAST_EXEC=$(echo "$LAST_EXEC_JSON" | grep -oE '[0-9]+' | head -1)

if [ -n "$LAST_EXEC" ] && [ "$LAST_EXEC" != "null" ] && [ "$LAST_EXEC" -gt 0 ]; then
    # Convert timestamp to readable date (timestamp is in milliseconds)
    LAST_SEC=$((LAST_EXEC / 1000))
    LAST_UTC=$(date -u -r $LAST_SEC +"%Y-%m-%d %H:%M:%S UTC" 2>/dev/null || date -u -d "@$LAST_SEC" +"%Y-%m-%d %H:%M:%S UTC" 2>/dev/null)
    LAST_LOCAL=$(date -r $LAST_SEC +"%Y-%m-%d %H:%M:%S %Z" 2>/dev/null || date -d "@$LAST_SEC" +"%Y-%m-%d %H:%M:%S %Z" 2>/dev/null)
    
    echo "📊 Last Execution:"
    if [ -n "$LAST_UTC" ]; then
        echo "   UTC: $LAST_UTC"
    fi
    if [ -n "$LAST_LOCAL" ]; then
        echo "   Local: $LAST_LOCAL"
    fi
    echo ""
    
    # Calculate next run (1 hour after last)
    NEXT_SEC=$((LAST_SEC + 3600))
    NOW_SEC=$(date +%s)
    
    if [ $NEXT_SEC -gt $NOW_SEC ]; then
        NEXT_UTC=$(date -u -r $NEXT_SEC +"%Y-%m-%d %H:%M:%S UTC" 2>/dev/null || date -u -d "@$NEXT_SEC" +"%Y-%m-%d %H:%M:%S UTC" 2>/dev/null)
        NEXT_LOCAL=$(date -r $NEXT_SEC +"%Y-%m-%d %H:%M:%S %Z" 2>/dev/null || date -d "@$NEXT_SEC" +"%Y-%m-%d %H:%M:%S %Z" 2>/dev/null)
        MINUTES_UNTIL=$(( ($NEXT_SEC - $NOW_SEC) / 60 ))
        
        echo "⏰ Next Scheduled Run:"
        if [ -n "$NEXT_UTC" ]; then
            echo "   UTC: $NEXT_UTC"
        fi
        if [ -n "$NEXT_LOCAL" ]; then
            echo "   Local: $NEXT_LOCAL"
        fi
        echo "   In: $MINUTES_UNTIL minutes"
    else
        echo "⏰ Next run should be soon (within the hour)"
    fi
else
    echo "📊 No recent executions found"
    echo "⏰ Next run will be within 1 hour"
fi

echo ""
echo "💡 Note: EventBridge 'rate(1 hour)' runs approximately every hour"
echo "   from when the rule was created, not at exact hour marks."

