#!/usr/bin/env python3
import boto3
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SageMakerMetricsLogger:
    """
    Logger for training metrics that integrates with CloudWatch Metrics
    (compatible with SageMaker monitoring and visualization).
    
    Usage:
        logger = SageMakerMetricsLogger(
            training_job_name="paper_123",
            paper_id="paper_123",
            instance_type="trn1.2xlarge"
        )
        
        # Log metrics
        logger.log_metric("training_loss", 0.5, step=1)
        logger.log_metric("validation_accuracy", 0.95, step=1)
        logger.log_metrics({"epoch": 1, "lr": 0.001})
    """
    
    def __init__(
        self,
        training_job_name: str,
        paper_id: Optional[str] = None,
        instance_type: str = "trn1.2xlarge",
        namespace: str = "Trainium/Training",
        region: str = "us-east-1"
    ):

        self.training_job_name = training_job_name
        self.paper_id = paper_id or training_job_name
        self.instance_type = instance_type
        self.namespace = namespace
        self.region = region
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        logger.info(f"Initialized SageMakerMetricsLogger for {training_job_name}")
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        unit: str = "None"
    ):

        if timestamp is None:
            timestamp = datetime.utcnow()
        
        metric_data = {
            'MetricName': metric_name,
            'Value': float(value),
            'Timestamp': timestamp,
            'Unit': unit,
            'Dimensions': [
                {
                    'Name': 'TrainingJobName',
                    'Value': self.training_job_name
                },
                {
                    'Name': 'PaperId',
                    'Value': self.paper_id
                },
                {
                    'Name': 'InstanceType',
                    'Value': self.instance_type
                }
            ]
        }
        
        if step is not None:
            metric_data['Dimensions'].append({
                'Name': 'Step',
                'Value': str(step)
            })
        
        self.metrics_buffer.append(metric_data)
        
        if len(self.metrics_buffer) >= 20:
            self.flush()
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):

        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, step=step, timestamp=timestamp)
    
    def log_execution_metrics(
        self,
        execution_time: float,
        success: bool,
        peak_memory_mb: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):

        # Log execution metrics
        self.log_metric("execution_time_seconds", execution_time, unit="Seconds")
        self.log_metric("execution_success", 1.0 if success else 0.0, unit="Count")
        
        if peak_memory_mb is not None:
            self.log_metric("peak_memory_mb", peak_memory_mb, unit="Megabytes")
        
        # Calculate cost (trn1.2xlarge is ~$1.34/hour)
        execution_hours = execution_time / 3600
        estimated_cost = execution_hours * 1.34
        self.log_metric("estimated_cost_usd", estimated_cost, unit="None")
        self.log_metric("execution_hours", execution_hours, unit="None")  # Hours not a valid CloudWatch unit
        
        # Log any additional metrics
        if additional_metrics:
            for metric_name, value in additional_metrics.items():
                self.log_metric(metric_name, value)
        
        # Flush all metrics
        self.flush()
    
    def flush(self):
        if not self.metrics_buffer:
            return
        
        try:
            # CloudWatch allows up to 20 metrics per request
            for i in range(0, len(self.metrics_buffer), 20):
                batch = self.metrics_buffer[i:i+20]
                
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=batch
                )
            
            logger.info(f"Flushed {len(self.metrics_buffer)} metrics to CloudWatch")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics to CloudWatch: {e}")
            # Don't raise - metrics logging failure shouldn't break training
    
    def extract_and_log_metrics_from_output(self, stdout: str):
        
        try:
            step = 0
            for line in stdout.split('\n'):
                if line.startswith('METRICS:'):
                    json_str = line.replace('METRICS:', '').strip()
                    try:
                        parsed_metrics = json.loads(json_str)
                        
                        # Extract step/epoch if present
                        if 'step' in parsed_metrics:
                            step = int(parsed_metrics.pop('step'))
                        elif 'epoch' in parsed_metrics:
                            step = int(parsed_metrics.pop('epoch'))
                        
                        # Log all metrics
                        self.log_metrics(parsed_metrics, step=step if step > 0 else None)
                        
                        # Increment step if not provided
                        if step == 0:
                            step += 1
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse METRICS line: {line[:100]}... Error: {e}")
            
            # Flush after parsing all metrics
            if self.metrics_buffer:
                self.flush()
                
        except Exception as e:
            logger.warning(f"Failed to extract metrics from output: {e}")


def create_metrics_logger(
    paper_id: str,
    paper_title: Optional[str] = None,
    instance_type: str = "trn1.2xlarge",
    namespace: str = "Trainium/Training"
) -> SageMakerMetricsLogger:

    training_job_name = f"{paper_id}_{int(time.time())}"
    
    logger_obj = SageMakerMetricsLogger(
        training_job_name=training_job_name,
        paper_id=paper_id,
        instance_type=instance_type,
        namespace=namespace
    )
    
    if paper_title:
        logger.info(f"Created metrics logger for: {paper_title} ({paper_id})")
    
    return logger_obj

