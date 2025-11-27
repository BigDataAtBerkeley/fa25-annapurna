#!/usr/bin/env python3
"""
Quick test script to verify Slack integration works.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add code-gen-for-deliv to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code-gen-for-deliv'))

from slack_notifier import SlackNotifier
from opensearch_client import OpenSearchClient

def test_slack():
    """Test Slack notification with a real paper."""
    
    # Check if Slack is configured
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_CHANNEL")
    
    if not bot_token:
        print("❌ SLACK_BOT_TOKEN not set in .env file")
        return
    
    if not channel:
        print("❌ SLACK_CHANNEL not set in .env file")
        return
    
    print(f"✅ Slack token found: {bot_token[:10]}...")
    print(f"✅ Slack channel: {channel}")
    
    # Test channel access first
    print("\n🔍 Testing channel access...")
    slack = SlackNotifier()
    if slack.send_simple_message("🧪 Testing channel access...", channel=channel):
        print("✅ Channel access works! Bot can send messages.")
    else:
        print("❌ Channel access failed. Common issues:")
        print("   1. Bot not invited to the channel")
        print("   2. Channel ID is incorrect")
        print("   3. Try using channel name like '#channel-name' instead")
        print("\n💡 To fix:")
        print("   - Invite the bot to the channel: /invite @YourBotName")
        print("   - Or use channel name: export SLACK_CHANNEL='#channel-name'")
        return
    
    # Get a paper from OpenSearch
    paper_id = input("\nEnter a paper ID to test (or press Enter to skip): ").strip()
    
    if not paper_id:
        print("Skipping paper test. Testing simple message instead...")
        slack = SlackNotifier()
        if slack.send_simple_message("🧪 Test message from Paper Pipeline Bot!"):
            print("✅ Slack message sent successfully!")
        else:
            print("❌ Failed to send Slack message")
        return
    
    try:
        # Get paper from OpenSearch
        opensearch = OpenSearchClient()
        paper = opensearch.get_paper_by_id(paper_id)
        
        if not paper:
            print(f"❌ Paper {paper_id} not found in OpenSearch")
            return
        
        print(f"✅ Found paper: {paper.get('title', 'Unknown')}")
        
        # Send to Slack
        slack = SlackNotifier()
        paper['_id'] = paper_id  # Add paper ID for formatting
        
        print("\n📤 Sending paper info to Slack...")
        if slack.send_paper_info(paper):
            print("✅ Paper info sent to Slack successfully!")
        else:
            print("❌ Failed to send paper info to Slack")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_slack()

