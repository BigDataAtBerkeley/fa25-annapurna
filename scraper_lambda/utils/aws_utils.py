import os
import json
import uuid
import boto3
from utils.logging_utils import setup_logger

logger = setup_logger("aws_utils")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("BUCKET_NAME")
QUEUE_URL = os.getenv("QUEUE_URL")

session = boto3.Session(region_name=AWS_REGION)
s3 = session.client("s3")
sqs = session.client("sqs")

def upload_pdf_to_s3(pdf_stream, key: str):
    s3.upload_fileobj(pdf_stream, BUCKET, key)
    logger.info(f"Uploaded PDF to s3://{BUCKET}/{key}")

def send_to_sqs(title, authors, date, abstract, key):
    msg = {
        "title": title,
        "authors": authors,
        "date": date,
        "abstract": abstract,
        "s3_bucket": BUCKET,
        "s3_key": key,
    }
    resp = sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(msg),
        MessageGroupId="research-parsing",
        MessageDeduplicationId=str(uuid.uuid4()),
    )
    logger.info(f"Sent SQS MessageId={resp['MessageId']} | Title={title}")
