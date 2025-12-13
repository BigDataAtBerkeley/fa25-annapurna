"""
Slack Notifier for Research Papers
Sends paper information from OpenSearch to a Slack channel
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
        
        abstract = paper.get('abstract', '')
        if abstract:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Abstract:*\n{abstract}"
                }
            })
        
        
        s3_bucket = paper.get('s3_bucket')
        s3_key = paper.get('s3_key')
        if s3_bucket and s3_key:

            s3_console_url = f"https://s3.console.aws.amazon.com/s3/object/{s3_bucket}/{s3_key}"
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📄 Paper PDF:*\n<{s3_console_url}|View in S3 Console>"
                }
            })
        
        return blocks
    
    def send_paper_info(self, paper: Dict[str, Any], channel: Optional[str] = None) -> Optional[str]:
        """
        Send paper information to Slack channel
        Returns: Thread timestamp (ts) if message sent successfully, None otherwise
        """
        if not self.enabled:
            logger.warning("Slack notifier is disabled (no bot token)")
            return False
        
        target_channel = channel or self.default_channel
        if not target_channel:
            logger.error("No Slack channel specified")
            return False
        
        try:
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
                thread_ts = result.get("ts")
                logger.info(f"Sent paper info to Slack channel {target_channel} (thread_ts: {thread_ts})")
                return thread_ts
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to send to Slack: {error}")
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
        Send initial code generation notification as follow-up in thread
        Returns: True if message sent successfully, False otherwise
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
            
            fields = []
            
            if recommended_dataset:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*Recommended Dataset:*\n{recommended_dataset}"
                })
            
            if model_used:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*Model:*\n{model_used}"
                })
            
            fields.append({
                "type": "mrkdwn",
                "text": f"*Code Length:*\n{code_length:,} characters"
            })
            

            for i in range(0, len(fields), 2):
                section_fields = fields[i:i+2]
                blocks.append({
                    "type": "section",
                    "fields": section_fields
                })
            
            # S3 link 
            if code_s3_key:
                code_s3_bucket = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
                # Create clickable S3 console URL (correct format for specific object)
                s3_console_url = f"https://s3.console.aws.amazon.com/s3/object/{code_s3_bucket}/{code_s3_key}"
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*📄 Code Location:*\n<{s3_console_url}|View in S3 Console>"
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
                logger.info(f"Sent code generation notification to Slack (thread: {thread_ts})")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to send to Slack: {error}")
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
        Returns:True if message sent successfully, False otherwise
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
            
            # Code S3 link - construct from code_s3_key or code_s3_location
            code_s3_key = paper.get('code_s3_key', '')
            code_s3_location = paper.get('code_s3_location', '')
            if code_s3_key or code_s3_location:

                if code_s3_location and code_s3_location.startswith('s3://'):
                    s3_path = code_s3_location[5:]  # Remove 's3://'
                    bucket, key = s3_path.split('/', 1) if '/' in s3_path else (s3_path, '')
                elif code_s3_key:
                    bucket = os.getenv('CODE_BUCKET', 'papers-code-artifacts')
                    key = code_s3_key
                else:
                    bucket = None
                    key = None
                
                if bucket and key:
                    s3_console_url = f"https://s3.console.aws.amazon.com/s3/object/{bucket}/{key}"
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*📄 Code Location:*\n<{s3_console_url}|View in S3 Console>"
                        }
                    })
            
            # Results S3 link 
            if paper.get('results_s3_location'):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*📊 Results Location:*\n`{paper['results_s3_location']}`"
                    }
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
        
            if thread_ts:
                payload["thread_ts"] = thread_ts
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("ok"):
                logger.info(f"Sent execution notification to Slack channel {target_channel}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to send to Slack: {error}")
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
                logger.info(f"Sent message to Slack channel {target_channel}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Failed to send to Slack: {error}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Slack: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending to Slack: {e}")
            return None

