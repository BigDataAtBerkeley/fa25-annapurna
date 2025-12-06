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

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
        
        # Exclude error fields from old runs (only show current/clean metadata)
        other_fields = {}
        known_fields = {'title', 'authors', 'abstract', '_id', 's3_bucket', 's3_key', 
                       'abstract_embedding', 'sha_abstract', 'title_normalized'}
        # Exclude error-related fields from initial message
        error_fields = {'has_errors', 'error_message', 'error_type', 'errors', 'tested', 
                       'code_generated', 'code_s3_key', 'code_s3_bucket', 'code_generated_at'}
        
        for key, value in paper.items():
            if key not in known_fields and key not in error_fields:
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
    
    def send_paper_info(self, paper: Dict[str, Any], channel: Optional[str] = None) -> Optional[str]:
        """
        Send paper information to Slack channel.
        
        Args:
            paper: Paper document from OpenSearch
            channel: Slack channel ID or name (default: use default_channel)
            
        Returns:
            Thread timestamp (ts) if message sent successfully, None otherwise
            This can be used to reply in the same thread later
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
                thread_ts = result.get("ts")  # Get message timestamp for threading
                logger.info(f"✅ Sent paper info to Slack channel {target_channel} (thread_ts: {thread_ts})")
                return thread_ts
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to send to Slack: {error}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Slack: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending to Slack: {e}")
            return None
    
    def send_code_generation_notification(self, paper_id: str, code_length: int, 
                                         model_used: Optional[str] = None, 
                                         recommended_dataset: Optional[str] = None,
                                         code_s3_key: Optional[str] = None,
                                         channel: Optional[str] = None, 
                                         thread_ts: Optional[str] = None) -> bool:
        """
        Send initial code generation notification as follow-up in thread.
        
        Args:
            paper_id: Paper ID
            code_length: Length of generated code
            model_used: Model used for generation (optional)
            recommended_dataset: Recommended dataset (optional)
            code_s3_key: S3 key for code file (optional)
            channel: Slack channel ID or name (default: use default_channel)
            thread_ts: Thread timestamp to reply in thread (required)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Slack notifier is disabled (no bot token)")
            return False
        
        if not thread_ts:
            logger.error("thread_ts is required for code generation notification")
            return False
        
        target_channel = channel or self.default_channel
        if not target_channel:
            logger.error("No Slack channel specified")
            return False
        
        try:
            blocks = []
            
            # Header
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "✅ Initial Code Generation Complete"
                }
            })
            
            blocks.append({"type": "divider"})
            
            # Code generation details
            fields = [
                {
                    "type": "mrkdwn",
                    "text": f"*Paper ID:*\n`{paper_id}`"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Code Length:*\n{code_length:,} characters"
                }
            ]
            
            if model_used:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*Model:*\n{model_used}"
                })
            
            if recommended_dataset:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*Recommended Dataset:*\n{recommended_dataset}"
                })
            
            # Split into sections of 2 fields each
            for i in range(0, len(fields), 2):
                section_fields = fields[i:i+2]
                blocks.append({
                    "type": "section",
                    "fields": section_fields
                })
            
            # S3 link if available
            if code_s3_key:
                code_s3_bucket = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
                aws_region = os.getenv('AWS_REGION', 'us-east-1')
                # Create clickable S3 console URL
                s3_console_url = f"https://s3.console.aws.amazon.com/s3/object/{code_s3_bucket}?prefix={code_s3_key}"
                # Also show the s3:// path for reference
                s3_path = f"s3://{code_s3_bucket}/{code_s3_key}"
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Code Location:*\n<{s3_console_url}|View in S3 Console>\n`{s3_path}`"
                    }
                })
            
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "📤 Code sent to Flask app for execution and code review"
                    }
                ]
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
                "text": f"Initial code generation complete for {paper_id}",
                "thread_ts": thread_ts
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("ok"):
                logger.info(f"✅ Sent code generation notification to Slack (thread: {thread_ts})")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to send to Slack: {error}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending code generation notification to Slack: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to Slack: {e}")
            return False
    
    def send_execution_notification(self, paper: Dict[str, Any], execution_result: Dict[str, Any], 
                                    channel: Optional[str] = None, thread_ts: Optional[str] = None) -> bool:
        """
        Send execution notification with paper info, execution results, and code links.
        
        Args:
            paper: Paper document (filtered, no embeddings)
            execution_result: Execution result dictionary
            channel: Slack channel ID or name (default: use default_channel)
            thread_ts: Thread timestamp to reply in thread (optional)
            
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
            
            # Add thread_ts if provided to reply in thread
            if thread_ts:
                payload["thread_ts"] = thread_ts
            
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
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Slack: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending to Slack: {e}")
            return None

