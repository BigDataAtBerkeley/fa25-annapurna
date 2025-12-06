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
        
        # Abstract (truncated to 800 chars for better visibility)
        abstract = paper.get('abstract', '')
        if abstract:
            abstract_preview = abstract[:800] + "..." if len(abstract) > 800 else abstract
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Abstract:*\n{abstract_preview}"
                }
            })
        
        # All other fields in a collapsible section
        other_fields = {}
        known_fields = {'title', 'authors', 'abstract', '_id', 's3_bucket', 's3_key', 
                       'abstract_embedding', 'sha_abstract', 'title_normalized'}
        
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
            aws_region = os.getenv('AWS_REGION', 'us-east-1')
            s3_console_url = f"https://s3.console.aws.amazon.com/s3/object/{s3_bucket}?prefix={s3_key}"
            s3_path = f"s3://{s3_bucket}/{s3_key}"
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*S3 Location:*\n<{s3_console_url}|View in S3 Console>\n`{s3_path}`"
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
    
    def send_final_code_notification(self, paper_id: str, code_length: int,
                                    code_review_iterations: int,
                                    code_s3_key: Optional[str] = None,
                                    channel: Optional[str] = None,
                                    thread_ts: Optional[str] = None) -> bool:
        """
        Send final code notification after code review (second follow-up).
        
        Args:
            paper_id: Paper ID
            code_length: Length of final code after code review
            code_review_iterations: Number of code review iterations
            code_s3_key: S3 key for final code file (optional)
            channel: Slack channel ID or name (default: use default_channel)
            thread_ts: Thread timestamp to reply in thread (required)
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Slack notifier is disabled (no bot token)")
            return False
        
        if not thread_ts:
            logger.error("thread_ts is required for final code notification")
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
                    "text": "🔧 Final Code After Code Review"
                }
            })
            
            blocks.append({"type": "divider"})
            
            # Code review details
            fields = [
                {
                    "type": "mrkdwn",
                    "text": f"*Paper ID:*\n`{paper_id}`"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Code Review Iterations:*\n{code_review_iterations}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Final Code Length:*\n{code_length:,} characters"
                }
            ]
            
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
                s3_console_url = f"https://s3.console.aws.amazon.com/s3/object/{code_s3_bucket}?prefix={code_s3_key}"
                s3_path = f"s3://{code_s3_bucket}/{code_s3_key}"
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Final Code Location:*\n<{s3_console_url}|View in S3 Console>\n`{s3_path}`"
                    }
                })
            
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "✅ Code reviewed and fixed. Running final execution..."
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
                "text": f"Final code after code review for {paper_id}",
                "thread_ts": thread_ts
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("ok"):
                logger.info(f"✅ Sent final code notification to Slack (thread: {thread_ts})")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to send to Slack: {error}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending final code notification to Slack: {e}")
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
            
            # Header with title and execution status (FINAL execution results)
            title = paper.get('title', 'Unknown Title')
            execution_status = paper.get('execution_status', 'UNKNOWN')
            status_emoji = '✅' if execution_result.get('success') else '❌'
            
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Final Execution Results: {title[:60]}{'...' if len(title) > 60 else ''}"
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
            
            # Abstract (truncated to 800 chars for better visibility)
            abstract = paper.get('abstract', '')
            if abstract:
                abstract_preview = abstract[:800] + "..." if len(abstract) > 800 else abstract
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
                # Show first 1500 chars of error message
                error_preview = error_msg[:1500] + "..." if len(error_msg) > 1500 else error_msg
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*❌ Error:*\n```{error_preview}```"
                    }
                })
                
                # Show more of stderr (1500 chars)
                if paper.get('execution_stderr_preview'):
                    stderr_preview = paper['execution_stderr_preview'][:1500] + "..." if len(paper['execution_stderr_preview']) > 1500 else paper['execution_stderr_preview']
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Stderr Preview:*\n```{stderr_preview}```"
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
            
            # Execution metrics (if available) - check both paper (from OpenSearch) and execution_result
            execution_metrics = paper.get('execution_metrics', {})
            # Also check execution_result for metrics (in case OpenSearch hasn't been updated yet)
            if not execution_metrics and execution_result:
                # Check for metrics in various possible locations
                execution_metrics = execution_result.get('metrics', {}) or execution_result.get('detailed_metrics', {})
                # Also check if metrics are directly in execution_result (from **metrics spread)
                if not execution_metrics:
                    # Extract common metric keys from execution_result
                    metric_keys = ['epoch', 'epochs', 'loss', 'training_loss', 'accuracy', 'val_accuracy', 'learning_rate', 'lr']
                    execution_metrics = {k: execution_result.get(k) for k in metric_keys if execution_result.get(k) is not None}
            
            if execution_metrics:
                metrics_text = "*📈 Training Metrics:*\n"
                # Extract common metrics
                metrics_to_show = []
                if 'epoch' in execution_metrics or 'epochs' in execution_metrics:
                    epochs = execution_metrics.get('epoch') or execution_metrics.get('epochs')
                    metrics_to_show.append(f"Epochs: {epochs}")
                if 'loss' in execution_metrics or 'training_loss' in execution_metrics:
                    loss = execution_metrics.get('loss') or execution_metrics.get('training_loss')
                    metrics_to_show.append(f"Loss: {loss:.4f}" if isinstance(loss, (int, float)) else f"Loss: {loss}")
                if 'accuracy' in execution_metrics or 'val_accuracy' in execution_metrics:
                    acc = execution_metrics.get('accuracy') or execution_metrics.get('val_accuracy')
                    metrics_to_show.append(f"Accuracy: {acc:.4f}" if isinstance(acc, (int, float)) else f"Accuracy: {acc}")
                if 'learning_rate' in execution_metrics or 'lr' in execution_metrics:
                    lr = execution_metrics.get('learning_rate') or execution_metrics.get('lr')
                    metrics_to_show.append(f"LR: {lr}")
                
                if metrics_to_show:
                    metrics_text += " • ".join(metrics_to_show)
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": metrics_text
                        }
                    })
            
            # Results S3 link (if available) - make it clickable
            # Check both paper (from OpenSearch) and execution_result
            results_s3_location = paper.get('results_s3_location', '') or execution_result.get('s3_results_key', '')
            if not results_s3_location and execution_result:
                # Construct from known bucket and paper_id
                results_bucket = os.getenv('RESULTS_BUCKET', 'trainium-execution-results')
                paper_id = paper.get('_id', '')
                if paper_id:
                    results_s3_location = f"s3://{results_bucket}/results/{paper_id}/execution_result.json"
            
            if results_s3_location:
                # Parse S3 location to create console URL
                if results_s3_location.startswith('s3://'):
                    s3_path = results_s3_location[5:]  # Remove 's3://'
                    bucket, key = s3_path.split('/', 1) if '/' in s3_path else (s3_path, '')
                    aws_region = os.getenv('AWS_REGION', 'us-east-1')
                    s3_console_url = f"https://s3.console.aws.amazon.com/s3/object/{bucket}?prefix={key}"
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*📊 Execution Results:*\n<{s3_console_url}|View in S3 Console>\n`{results_s3_location}`\n*Contains:* Metrics, epochs, loss, stdout/stderr, and execution details"
                        }
                    })
                else:
                    # Fallback if not in s3:// format
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*📊 Execution Results:*\n`{results_s3_location}`"
                        }
                    })
            
            # Profiler information (if available) - make it clickable
            profiler_info = execution_result.get('profiler', {}) if execution_result else {}
            if paper.get('profiler_enabled') or profiler_info.get('profiler_enabled'):
                profiler_files = paper.get('profiler_files', []) or profiler_info.get('profiler_files', [])
                profiler_s3 = paper.get('profiler_s3_location', '')
                perfetto_file = paper.get('profiler_perfetto_file', '') or profiler_info.get('perfetto_file', '')
                
                # Construct profiler S3 location if not available
                if not profiler_s3:
                    results_bucket = os.getenv('RESULTS_BUCKET', 'trainium-execution-results')
                    paper_id = paper.get('_id', '')
                    if paper_id:
                        profiler_s3 = f"s3://{results_bucket}/profiler/{paper_id}/"
                
                profiler_text = f"*🔬 Profiler Artifacts*\n"
                if profiler_s3:
                    # Parse S3 location to create console URL
                    if profiler_s3.startswith('s3://'):
                        s3_path = profiler_s3[5:]  # Remove 's3://'
                        bucket, key_prefix = s3_path.split('/', 1) if '/' in s3_path else (s3_path, '')
                        aws_region = os.getenv('AWS_REGION', 'us-east-1')
                        s3_console_url = f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}?prefix={key_prefix}"
                        profiler_text += f"<{s3_console_url}|View in S3 Console>\n`{profiler_s3}`\n"
                    else:
                        profiler_text += f"S3 Location: `{profiler_s3}`\n"
                if perfetto_file:
                    profiler_text += f"Perfetto File: `{perfetto_file}`\n"
                if profiler_files:
                    profiler_text += f"*Files:* {len(profiler_files)} profiler files (including .ptrace files)"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": profiler_text
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

