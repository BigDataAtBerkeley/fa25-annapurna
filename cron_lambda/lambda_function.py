"""
Cron Lambda for Pipeline Automation

This Lambda runs every 1 hour and:
1. Queries OpenSearch for papers without executed_on_trn = true
2. Limits processing to MAX_PAPERS_PER_RUN (default 10) papers per run
3. Sends ALL papers to code-evaluation.fifo queue for code regeneration
   - If executed_on_trn != true, code is regenerated and paper goes through entire process again
   - This ensures failed executions get fresh code and restart the pipeline
4. Code generation lambda handles sending papers to Trainium after code is generated
5. Remaining papers wait for the next cron job run
6. Auto manages Trainium EC2 instance (starts when papers waiting, stops when idle)
7. Maintains Slack thread continuity
"""

import os
import json
import logging
import time
import boto3
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info("Cron Lambda started")

# Environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "research-papers-v3")
CODE_EVAL_QUEUE_URL = os.getenv("CODE_EVAL_QUEUE_URL")  # SQS queue for code generation
TRAINIUM_ENDPOINT = os.getenv("TRAINIUM_ENDPOINT")  # Flask app endpoint for execution
MAX_CODE_GEN_CONCURRENT = int(os.getenv("MAX_CODE_GEN_CONCURRENT", "5"))  # Max concurrent code gen requests
MAX_TRAINIUM_CONCURRENT = int(os.getenv("MAX_TRAINIUM_CONCURRENT", "1"))  # Max concurrent trainium executions (how many papers can run simultaneously)
MAX_PAPERS_PER_RUN = int(os.getenv("MAX_PAPERS_PER_RUN", "1"))  # Maximum papers to process per cron job run (matches Lambda batch size)
TRAINIUM_INSTANCE_ID = os.getenv("TRAINIUM_INSTANCE_ID")  # EC2 instance ID for auto start/stop
TRAINIUM_EXECUTION_QUEUE_URL = os.getenv("TRAINIUM_EXECUTION_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo")  # SQS queue for execution

# AWS Clients
session = boto3.Session(region_name=AWS_REGION)
creds = session.get_credentials().get_frozen_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, "es")
sqs_client = boto3.client("sqs", region_name=AWS_REGION)
lambda_client = boto3.client("lambda", region_name=AWS_REGION)
ec2_client = boto3.client("ec2", region_name=AWS_REGION) if TRAINIUM_INSTANCE_ID else None

os_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://","").replace("http://",""), "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30,  # Reduced timeout for faster failure, will retry if needed
    max_retries=3,
    retry_on_timeout=True
)


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific paper by ID from OpenSearch.
    
    Args:
        paper_id: The paper ID to retrieve
        
    Returns:
        Paper data if found, None otherwise
    """
    try:
        response = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
        if response and response.get('found'):
            paper = response['_source']
            paper['_id'] = response['_id']
            logger.info(f"Found paper {paper_id}: {paper.get('title', 'Unknown title')}")
            return paper
        else:
            logger.warning(f"Paper {paper_id} not found in OpenSearch")
            return None
    except Exception as e:
        logger.error(f"Error getting paper {paper_id} from OpenSearch: {e}")
        return None


def get_papers_without_execution(size: int = 3) -> List[Dict[str, Any]]:
    """
    Query OpenSearch for papers that don't have executed_on_trn = true.
    
    Returns papers that either:
    - Don't have executed_on_trn field
    - Have executed_on_trn = false
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Query for papers where executed_on_trn is not true
            # This includes papers where the field is missing, false, or any value other than true
            query = {
                "bool": {
                    "must_not": {
                        "term": {"executed_on_trn": True}
                    }
                }
            }
            
            response = os_client.search(
                index=OPENSEARCH_INDEX,
                body={
                    "query": query,
                    "size": size,
                    "sort": [{"ingested_at": {"order": "asc"}}]  # Process oldest first
                },
                request_timeout=30  # 30 second timeout per request
            )
            
            papers = []
            for hit in response.get('hits', {}).get('hits', []):
                paper = hit['_source']
                paper['_id'] = hit['_id']
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers without executed_on_trn = true")
            return papers
            
        except Exception as e:
            error_str = str(e)
            if "timeout" in error_str.lower() or "Timeout" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    logger.warning(f"OpenSearch timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"OpenSearch timeout after {max_retries} attempts")
                    return []
            else: 
                logger.error(f"Error querying OpenSearch: {e}")
                return []
    
    return []


def check_code_gen_concurrency() -> int:
    """
    Check how many code generation lambdas are currently running.
    Returns the number of concurrent executions.
    """
    try:
        # Get the code gen lambda function name from environment or use default
        code_gen_lambda_name = os.getenv("CODE_GEN_LAMBDA_NAME", "PapersCodeGenerator-container")
        
        # Get concurrent executions metric from CloudWatch
        cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)
        
        # Get metric statistics for concurrent executions
        response = cloudwatch.get_metric_statistics(
            Namespace="AWS/Lambda",
            MetricName="ConcurrentExecutions",
            Dimensions=[
                {"Name": "FunctionName", "Value": code_gen_lambda_name}
            ],
            StartTime=datetime.utcnow().replace(second=0, microsecond=0),
            EndTime=datetime.utcnow(),
            Period=60,
            Statistics=["Maximum"]
        )
        
        if response.get("Datapoints"):
            concurrent = int(response["Datapoints"][0]["Maximum"])
            logger.info(f"Code gen lambda concurrent executions: {concurrent}")
            return concurrent
        else:
            # If no metric available, assume we can proceed (conservative)
            logger.warning("Could not get concurrent executions metric, assuming safe")
            return 0
            
    except Exception as e:
        logger.warning(f"Error checking code gen concurrency: {e}")
        return 0  # Conservative: assume no concurrent executions


def check_trainium_availability() -> bool:
    """
    Check if trainium instance is available and running.
    Returns True if trainium is available, False otherwise.
    """
    if not TRAINIUM_ENDPOINT:
        logger.warning("TRAINIUM_ENDPOINT not set - cannot check trainium availability")
        return False
    
    try:
        # Check health endpoint
        health_url = f"{TRAINIUM_ENDPOINT.rstrip('/')}/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            logger.info("✅ Trainium is available and healthy")
            return True
        else:
            logger.warning(f"⚠️ Trainium health check returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ Trainium not available: {e}")
        return False


def check_trainium_concurrency() -> int:
    """
    Check how many executions are currently running on trainium.
    Returns the number of concurrent executions.
    """
    if not TRAINIUM_ENDPOINT:
        return 0
    
    try:
        # Check status endpoint if available
        status_url = f"{TRAINIUM_ENDPOINT.rstrip('/')}/status"
        response = requests.get(status_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            running = data.get("running_executions", 0)
            logger.info(f"Trainium running executions: {running}")
            return running
        else:
            # If status endpoint not available, assume we can check health
            if check_trainium_availability():
                return 0  # Assume no running executions if healthy
            return MAX_TRAINIUM_CONCURRENT  # Assume at capacity if unhealthy
            
    except Exception as e:
        logger.warning(f"Error checking trainium concurrency: {e}")
        # If we can't check, assume it's available if health check passes
        if check_trainium_availability():
            return 0
        return MAX_TRAINIUM_CONCURRENT


def send_to_code_eval_queue(paper_id: str, paper_data: Dict) -> bool:
    """
    Send paper to code-evaluation queue for code generation.
    
    Args:
        paper_id: OpenSearch document ID
        paper_data: Paper metadata
        
    Returns:
        True if successful, False otherwise
    """
    if not CODE_EVAL_QUEUE_URL:
        logger.warning("CODE_EVAL_QUEUE_URL not set, skipping queue")
        return False
    
    try:
        message = {
            "paper_id": paper_id,
            "action": "generate_by_id",
            "paper_title": paper_data.get("title"),
            "queued_at": datetime.now().isoformat()
        }
        
        sqs_client.send_message(
            QueueUrl=CODE_EVAL_QUEUE_URL,
            MessageBody=json.dumps(message),
            MessageGroupId=paper_id,
            MessageDeduplicationId=f"{paper_id}-{int(time.time() * 1000)}"
        )
        
        logger.info(f"Sent to code-eval queue: {paper_data.get('title')} ({paper_id})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send to code-eval queue: {e}")
        return False


def send_to_trainium(paper_id: str, code_s3_key: str, paper_title: Optional[str] = None) -> bool:
    """
    Send paper with code to trainium for execution.
    
    Args:
        paper_id: Paper ID
        code_s3_key: S3 key for the code file
        paper_title: Paper title (optional)
        
    Returns:
        True if successful, False otherwise
    """
    if not TRAINIUM_ENDPOINT:
        logger.warning("TRAINIUM_ENDPOINT not set, cannot send to trainium")
        return False
    
    try:
        # Get code from S3
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        code_bucket = os.getenv("CODE_BUCKET", "papers-code-artifacts")
        
        try:
            response = s3_client.get_object(Bucket=code_bucket, Key=code_s3_key)
            code = response['Body'].read().decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get code from S3 ({code_s3_key}): {e}")
            return False
        
        # Send to trainium Flask app
        endpoint = f"{TRAINIUM_ENDPOINT.rstrip('/')}/execute"
        payload = {
            "paper_id": paper_id,
            "code": code,
            "timeout": int(os.getenv("TRAINIUM_EXECUTION_TIMEOUT", "3600")),
            "paper_title": paper_title
        }
        
        # Get slack_thread_ts from OpenSearch if available (for thread continuity)
        try:
            response = os_client.get(index=OPENSEARCH_INDEX, id=paper_id)
            if response and response.get('_source', {}).get('slack_thread_ts'):
                payload["slack_thread_ts"] = response['_source']['slack_thread_ts']
                logger.info(f"Retrieved slack_thread_ts from OpenSearch for {paper_id}")
        except Exception as e:
            logger.debug(f"Could not get slack_thread_ts from OpenSearch (optional): {e}")
            pass  # Slack thread_ts is optional
        
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        
        logger.info(f"✅ Sent to trainium: {paper_title} ({paper_id})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send to trainium: {e}")
        return False


def get_papers_waiting_for_execution() -> int:
    """
    Count papers that are waiting for execution.
    Checks both:
    1. SQS queue (papers enqueued but not yet processed) - this is the most important check
    2. OpenSearch (papers with code but executed_on_trn != true)
    
    Returns:
        Number of papers waiting for execution
    """
    queue_count = 0
    opensearch_count = 0
    
    # Check SQS queue for papers waiting (most important - if queue has messages, instance should be running)
    try:
        trainium_queue_url = os.getenv("TRAINIUM_EXECUTION_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/478852001205/trainium-execution.fifo")
        if trainium_queue_url:
            response = sqs_client.get_queue_attributes(
                QueueUrl=trainium_queue_url,
                AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
            )
            attrs = response.get('Attributes', {})
            visible = int(attrs.get('ApproximateNumberOfMessages', 0))
            in_flight = int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0))
            queue_count = visible + in_flight
            if queue_count > 0:
                logger.info(f"Found {queue_count} papers in trainium-execution queue ({visible} visible, {in_flight} in flight)")
    except Exception as e:
        logger.warning(f"Error checking trainium-execution queue: {e}")
    
    # Check OpenSearch for papers with code but not executed (backup check)
    try:
        query = {
            "bool": {
                "must": [
                    {"exists": {"field": "code_s3_key"}}  # Has code
                ],
                "must_not": {
                    "term": {"executed_on_trn": True}  # Not executed yet
                }
            }
        }
        
        response = os_client.search(
            index=OPENSEARCH_INDEX,
            body={
                "query": query,
                "size": 0,  # We only need the count
                "track_total_hits": True
            },
            request_timeout=30
        )
        
        total = response.get('hits', {}).get('total', {})
        if isinstance(total, dict):
            opensearch_count = total.get('value', 0)
        else:
            opensearch_count = total if total else 0
        
        if opensearch_count > 0:
            logger.info(f"Found {opensearch_count} papers in OpenSearch waiting for execution (have code but not executed)")
    except Exception as e:
        logger.warning(f"Error counting papers in OpenSearch: {e}")
    
    total_waiting = queue_count + opensearch_count
    logger.info(f"Total papers waiting for execution: {total_waiting} ({queue_count} in queue, {opensearch_count} in OpenSearch)")
    return total_waiting


def get_trainium_instance_state() -> Optional[str]:
    """
    Get the current state of the Trainium EC2 instance.
    
    Returns:
        Instance state ('running', 'stopped', 'stopping', 'starting', etc.) or None if error
    """
    if not TRAINIUM_INSTANCE_ID or not ec2_client:
        return None
    
    try:
        response = ec2_client.describe_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
        if not response.get('Reservations'):
            return None
        
        instance = response['Reservations'][0]['Instances'][0]
        state = instance['State']['Name']
        logger.info(f"Trainium instance {TRAINIUM_INSTANCE_ID} state: {state}")
        return state
        
    except Exception as e:
        logger.error(f"Error getting Trainium instance state: {e}")
        return None


def start_trainium_instance() -> bool:
    """
    Start the Trainium EC2 instance if it's stopped.
    
    Returns:
        True if instance is running or was successfully started, False otherwise
    """
    if not TRAINIUM_INSTANCE_ID or not ec2_client:
        logger.warning("TRAINIUM_INSTANCE_ID not set - cannot manage instance")
        return False
    
    try:
        state = get_trainium_instance_state()
        
        if state == 'running':
            logger.info(f"✅ Trainium instance {TRAINIUM_INSTANCE_ID} is already running")
            return True
        elif state == 'stopped':
            logger.info(f"⏳ Starting Trainium instance {TRAINIUM_INSTANCE_ID}...")
            ec2_client.start_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
            # Wait for instance to be running (with timeout)
            logger.info("⏳ Waiting for instance to start...")
            waiter = ec2_client.get_waiter('instance_running')
            waiter.wait(
                InstanceIds=[TRAINIUM_INSTANCE_ID],
                WaiterConfig={'Delay': 15, 'MaxAttempts': 20}  # Max 5 minutes wait
            )
            
            logger.info(f"✅ Trainium instance {TRAINIUM_INSTANCE_ID} started successfully")
            
            # Wait a bit for services to start
            logger.info("⏳ Waiting 30 seconds for Trainium services to initialize...")
            time.sleep(30)
            
            return True
        elif state in ['starting', 'pending']:
            logger.info(f"⏳ Trainium instance {TRAINIUM_INSTANCE_ID} is already starting...")
            return True
        else:
            logger.warning(f"⚠️ Trainium instance {TRAINIUM_INSTANCE_ID} is in state: {state} - cannot start")
            return False
            
    except Exception as e:
        logger.error(f"Error starting Trainium instance: {e}")
        return False


def stop_trainium_instance() -> bool:
    """
    Stop the Trainium EC2 instance if it's running and no papers are waiting.
    
    Returns:
        True if instance was stopped or is already stopped, False otherwise
    """
    if not TRAINIUM_INSTANCE_ID or not ec2_client:
        logger.warning("TRAINIUM_INSTANCE_ID not set - cannot manage instance")
        return False
    
    try:
        state = get_trainium_instance_state()
        
        if state == 'stopped':
            logger.info(f"✅ Trainium instance {TRAINIUM_INSTANCE_ID} is already stopped")
            return True
        elif state == 'running':
            # Check if there are any running executions
            trainium_concurrent = check_trainium_concurrency()
            if trainium_concurrent > 0:
                logger.info(f"⚠️ Cannot stop Trainium instance - {trainium_concurrent} executions still running")
                return False
            
            logger.info(f"⏳ Stopping Trainium instance {TRAINIUM_INSTANCE_ID}...")
            ec2_client.stop_instances(InstanceIds=[TRAINIUM_INSTANCE_ID])
            
            # Wait for instance to stop (with timeout)
            logger.info("⏳ Waiting for instance to stop...")
            waiter = ec2_client.get_waiter('instance_stopped')
            waiter.wait(
                InstanceIds=[TRAINIUM_INSTANCE_ID],
                WaiterConfig={'Delay': 15, 'MaxAttempts': 20}  # Max 5 minutes wait
            )
            
            logger.info(f"✅ Trainium instance {TRAINIUM_INSTANCE_ID} stopped successfully")
            return True
        elif state in ['stopping']:
            logger.info(f"⏳ Trainium instance {TRAINIUM_INSTANCE_ID} is already stopping...")
            return True
        else:
            logger.warning(f"⚠️ Trainium instance {TRAINIUM_INSTANCE_ID} is in state: {state} - cannot stop")
            return False
            
    except Exception as e:
        logger.error(f"Error stopping Trainium instance: {e}")
        return False


def update_executed_on_trn(paper_id: str, value: bool = True) -> bool:
    """
    Update the executed_on_trn field in OpenSearch.
    
    Args:
        paper_id: Paper ID
        value: Value to set (default True)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os_client.update(
            index=OPENSEARCH_INDEX,
            id=paper_id,
            body={
                "doc": {
                    "executed_on_trn": value,
                    "executed_on_trn_updated_at": datetime.now().isoformat()
                }
            }
        )
        logger.info(f"Updated executed_on_trn={value} for paper {paper_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to update executed_on_trn for {paper_id}: {e}")
        return False


def lambda_handler(event, context):
    """
    Main Lambda handler for cron job.
    
    This function:
    1. Queries OpenSearch for papers without executed_on_trn = true
    2. Separates papers into two groups:
       - Papers without code: sends to code-evaluation.fifo
       - Papers with code: batches and sends to trainium
    3. Respects concurrency limits
    
    Event parameters:
    - paper_id (optional): If provided, process only this specific paper
    """
    logger.info("Starting cron job execution")
    
    try:
        # Check if a specific paper_id was provided in the event
        paper_id = None
        if isinstance(event, dict):
            paper_id = event.get('paper_id')
        elif isinstance(event, str):
            # Try to parse as JSON if it's a string
            try:
                event_dict = json.loads(event)
                paper_id = event_dict.get('paper_id')
            except:
                pass
        
        # Step 0: Manage Trainium instance based on queue status
        papers_waiting = get_papers_waiting_for_execution()
        
        if papers_waiting > 0:
            # There are papers waiting - ensure instance is running
            logger.info(f"Found {papers_waiting} papers waiting for execution - ensuring Trainium instance is running")
            start_trainium_instance()
        else:
            # No papers waiting - check if we should stop the instance
            trainium_concurrent = check_trainium_concurrency()
            if trainium_concurrent == 0:
                # No running executions and no papers waiting - stop instance to save costs
                logger.info("No papers waiting and no running executions - stopping Trainium instance to save costs")
                stop_trainium_instance()
        
        # Step 1: Get papers to process
        if paper_id:
            # Process specific paper
            logger.info(f"Processing specific paper: {paper_id}")
            paper = get_paper_by_id(paper_id)
            if not paper:
                return {
                    "statusCode": 404,
                    "body": json.dumps({
                        "message": f"Paper {paper_id} not found",
                        "paper_id": paper_id
                    })
                }
            papers_to_process = [paper]
        else:
            # Get papers without execution (normal cron behavior)
            papers_to_process = get_papers_without_execution(size=MAX_PAPERS_PER_RUN)
        
        if not papers_to_process:
            logger.info("No papers found without executed_on_trn = true")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No papers to process",
                    "papers_checked": 0
                })
            }
        
        logger.info(f"Executing {len(papers_to_process)}.")
        
        # Step 3: Send ALL papers to code-evaluation queue for code regeneration
        # If executed_on_trn != true, regenerate code and restart the entire process
        # This ensures papers that failed execution get fresh code and go through the process again
        code_gen_sent = 0
        
        logger.info(f"Sending {len(papers_to_process)} papers to code generation (regenerating code for all papers with executed_on_trn != true)")
        
        for paper in papers_to_process:
            paper_id = paper.get('_id')
            if send_to_code_eval_queue(paper_id, paper):
                code_gen_sent += 1
        
        # Step 4: No direct execution - papers will be executed after code generation completes
        # The code generation lambda will send papers to Trainium after generating code
        trainium_sent = 0
        trainium_concurrent = None
        trainium_available = check_trainium_availability()
        
        if trainium_available:
            trainium_concurrent = check_trainium_concurrency()
            logger.info(f"Trainium available: {trainium_available}, Concurrent: {trainium_concurrent}")
        else:
            logger.info("Trainium not available - code generation will handle execution when Trainium is ready")
        
        # Step 6: Final check - stop instance if no papers waiting and no executions running
        final_papers_waiting = get_papers_waiting_for_execution()
        final_trainium_concurrent = check_trainium_concurrency()
        
        if final_papers_waiting == 0 and final_trainium_concurrent == 0:
            logger.info("No papers waiting and no executions running - stopping Trainium instance")
            stop_trainium_instance()
        
        # Return summary
        result = {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Cron job completed",
                "paper_id": paper_id if paper_id else None,
                "papers_processed": len(papers_to_process),
                "code_gen_sent": code_gen_sent,
                "trainium_sent": trainium_sent,
                "trainium_concurrent": trainium_concurrent if trainium_available else None,
                "trainium_available": trainium_available,
                "papers_waiting_for_execution": papers_waiting,
                "trainium_instance_managed": TRAINIUM_INSTANCE_ID is not None,
                "note": "All papers sent to code generation - code will be regenerated and papers will go through full process again"
            })
        }
        
        logger.info(f"Cron job completed: {result['body']}")
        return result
        
    except Exception as e:
        logger.error(f"Error in cron job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }

