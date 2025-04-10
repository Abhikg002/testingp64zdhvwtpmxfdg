import boto3
import json
import time
import botocore.exceptions
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION

# Initialize AWS Bedrock client
bedrock_client = boto3.client("bedrock-runtime",
                              aws_access_key_id=AWS_ACCESS_KEY,
                              aws_secret_access_key=AWS_SECRET_KEY,
                              region_name=AWS_REGION)

def generate_embeddings(text, retries=3):
    """Convert text into embeddings using Titan Embeddings G1 - Text with retry logic."""
    for attempt in range(retries):
        try:
            response = bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]
        except botocore.exceptions.ClientError as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"[Titan] Retry {attempt+1}/{retries}: waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

