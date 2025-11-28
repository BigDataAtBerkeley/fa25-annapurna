"""
Slack Notifier for Research Papers

Sends paper information from OpenSearch to a Slack channel.
Uses Slack Web API (chat.postMessage) to send formatted messages.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SlackNotifier:
    """Send paper information to Slack channels."""
    
    def __init__(self, bot_token: Optional[str] = None, channel: Optional[str] = None):
        """
        Initialize Slack notifier.
        
        Args:
            bot_token: Slack bot token (default: from SLACK_BOT_TOKEN env var)
            channel: Default Slack channel ID or name (default: from SLACK_CHANNEL env var)
        """
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.default_channel = channel or os.getenv("SLACK_CHANNEL")
        
        if not self.bot_token:
            logger.warning("SLACK_BOT_TOKEN not set - Slack notifications will be disabled")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Slack notifier initialized")
        
        if not self.default_channel:
            logger.warning("SLACK_CHANNEL not set - must provide channel when sending messages")
    
    def _format_paper_fields(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format paper fields from OpenSearch into Slack Block Kit format.
        
        Args:
            paper: Paper document from OpenSearch
            
        Returns:
            List of Slack Block Kit blocks
        """
        blocks = []
        
        # Header with title
        title = paper.get('title', 'Unknown Title')
        paper_id = paper.get('_id', 'Unknown ID')
        
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📄 {title[:100]}{'...' if len(title) > 100 else ''}"
            }
        })
        
        # Divider
        blocks.append({"type": "divider"})
        
        # Paper ID
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Paper ID:*\n`{paper_id}`"
                }
            ]
        })
        
        # Authors
        authors = paper.get('authors', [])
        if authors:
            authors_text = ", ".join(authors[:5])  # Limit to first 5 authors
            if len(authors) > 5:
                authors_text += f" (+{len(authors) - 5} more)"
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Authors:*\n{authors_text}"
                    }
                ]
            })
        
        # Abstract (truncated)
        abstract = paper.get('abstract', '')
        if abstract:
            abstract_preview = abstract[:500] + "..." if len(abstract) > 500 else abstract
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Abstract:*\n{abstract_preview}"
                }
            })
        
        # All other fields in a collapsible section
        other_fields = {}
        known_fields = {'title', 'authors', 'abstract', '_id', 's3_bucket', 's3_key'}
        
        for key, value in paper.items():
            if key not in known_fields:
                # Format the value
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=2)[:200]
                    if len(json.dumps(value, indent=2)) > 200:
                        value_str += "..."
                elif isinstance(value, str) and len(value) > 200:
                    value_str = value[:200] + "..."
                else:
                    value_str = str(value)
                
                other_fields[key] = value_str
        
        if other_fields:

            fields_list = []
            for key, value in other_fields.items():
                fields_list.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}"
                })
            
            # Split into sections of 2 fields each (Slack limit per section)
            for i in range(0, len(fields_list), 2):
                section_fields = fields_list[i:i+2]
                blocks.append({
                    "type": "section",
                    "fields": section_fields
                })
        
        # S3 information if available
        s3_bucket = paper.get('s3_bucket')
        s3_key = paper.get('s3_key')
        if s3_bucket and s3_key:
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*S3 Location:*\n`s3://{s3_bucket}/{s3_key}`"
                    }
                ]
            })
        
        return blocks
    
    def send_paper_info(self, paper: Dict[str, Any], channel: Optional[str] = None) -> bool:
        """
        Send paper information to Slack channel.
        
        Args:
            paper: Paper document from OpenSearch
            channel: Slack channel ID or name (default: use default_channel)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Slack notifier is disabled (no bot token)")
            return False
        
        target_channel = channel or self.default_channel
        if not target_channel:
            logger.error("No Slack channel specified")
            return False
        
        try:
            # Format paper into Slack blocks
            blocks = self._format_paper_fields(paper)
            
            # Send message to Slack
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "channel": target_channel,
                "blocks": blocks,
                "text": f"Paper: {paper.get('title', 'Unknown')}"  # Fallback text
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("ok"):
                logger.info(f"✅ Sent paper info to Slack channel {target_channel}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to send to Slack: {error}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Slack: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to Slack: {e}")
            return False
    
    def send_execution_notification(self, paper: Dict[str, Any], execution_result: Dict[str, Any], 
                                    channel: Optional[str] = None) -> bool:
        """
        Send execution notification with paper info, execution results, and code links.
        
        Args:
            paper: Paper document (filtered, no embeddings)
            execution_result: Execution result dictionary
            channel: Slack channel ID or name (default: use default_channel)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Slack notifier is disabled (no bot token)")
            return False
        
        target_channel = channel or self.default_channel
        if not target_channel:
            logger.error("No Slack channel specified")
            return False
        
        try:
            blocks = []
            
            # Header with title and execution status
            title = paper.get('title', 'Unknown Title')
            execution_status = paper.get('execution_status', 'UNKNOWN')
            status_emoji = '✅' if execution_result.get('success') else '❌'
            
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} {title[:80]}{'...' if len(title) > 80 else ''}"
                }
            })
            
            blocks.append({"type": "divider"})
            
            # Paper ID
            paper_id = paper.get('_id', 'Unknown ID')
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Paper ID:*\n`{paper_id}`"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{execution_status}"
                    }
                ]
            })
            
            # Execution details
            exec_time = paper.get('execution_time_seconds', 0)
            return_code = paper.get('execution_return_code', -1)
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Execution Time:*\n{exec_time}s"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Return Code:*\n{return_code}"
                    }
                ]
            })
            
            # Authors
            authors = paper.get('authors', [])
            if authors:
                authors_text = ", ".join(authors[:3])  # Limit to first 3 authors
                if len(authors) > 3:
                    authors_text += f" (+{len(authors) - 3} more)"
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Authors:* {authors_text}"
                    }
                })
            
            # Abstract (truncated)
            abstract = paper.get('abstract', '')
            if abstract:
                abstract_preview = abstract[:300] + "..." if len(abstract) > 300 else abstract
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Abstract:*\n{abstract_preview}"
                    }
                })
            
            # Execution error details (if failed)
            if not execution_result.get('success'):
                error_msg = paper.get('execution_error', 'Unknown error')
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*❌ Error:*\n```{error_msg[:500]}{'...' if len(error_msg) > 500 else ''}```"
                    }
                })
                
                if paper.get('execution_stderr_preview'):
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Stderr Preview:*\n```{paper['execution_stderr_preview'][:500]}```"
                        }
                    })
            else:
                # Success - show stdout preview if available
                if paper.get('execution_stdout_preview'):
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*✅ Output Preview:*\n```{paper['execution_stdout_preview'][:500]}```"
                        }
                    })
            
            # Code S3 link
            code_s3_location = paper.get('code_s3_location', '')
            if code_s3_location:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*📄 Code Location:*\n<{paper.get('code_s3_url', '')}|{code_s3_location}>"
                    }
                })
            
            # Results S3 link (if available)
            if paper.get('results_s3_location'):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*📊 Results Location:*\n`{paper['results_s3_location']}`"
                    }
                })
            
            # Additional paper metadata
            if paper.get('venue'):
                blocks.append({
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Venue:*\n{paper['venue']}"
                        }
                    ]
                })
            
            if paper.get('url'):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*🔗 Paper URL:*\n<{paper['url']}|View Paper>"
                    }
                })
            
            # Send message to Slack
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "channel": target_channel,
                "blocks": blocks,
                "text": f"Execution {execution_status}: {title}"  # Fallback text
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("ok"):
                logger.info(f"✅ Sent execution notification to Slack channel {target_channel}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to send to Slack: {error}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending execution notification to Slack: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to Slack: {e}")
            return False
    
    def send_simple_message(self, message: str, channel: Optional[str] = None) -> bool:
        """
        Send a simple text message to Slack.
        
        Args:
            message: Message text
            channel: Slack channel ID or name (default: use default_channel)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Slack notifier is disabled (no bot token)")
            return False
        
        target_channel = channel or self.default_channel
        if not target_channel:
            logger.error("No Slack channel specified")
            return False
        
        try:
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "channel": target_channel,
                "text": message
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("ok"):
                logger.info(f"✅ Sent message to Slack channel {target_channel}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to send to Slack: {error}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Slack: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to Slack: {e}")
            return False

