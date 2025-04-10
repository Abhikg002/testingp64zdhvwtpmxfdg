import boto3
import json
import re
import time
import botocore.exceptions
from src.logger import logger
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION


def run_skill_extraction_prompt(text, retries=3):
    """
    Calls Anthropic Claude v2 via Bedrock to extract technical skills
    from a given resume or job description text.

    Returns:
        List[str]: Cleaned list of skill strings (lowercase, no duplicates).
    """
    #client = boto3.client(service_name='bedrock-runtime')

    from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION

    client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    model_id = 'anthropic.claude-v2'
    accept = 'application/json'
    content_type = 'application/json'

    prompt = f"""
Human: Extract only the technical skills from the following text. These should include programming languages, frameworks, libraries, software tools, cloud platforms, and technical certifications. Return the skills as a valid Python list of strings. Do not include soft skills, company names, locations, or general strengths.

Text:
\"\"\"{text}\"\"\"

Assistant:"""

    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 1000,
        "temperature": 0.2,
        "top_p": 0.9,
    })

    for attempt in range(retries):
        try:
            response = client.invoke_model(
                body=body,
                modelId=model_id,
                accept=accept,
                contentType=content_type,
            )
            response_body = json.loads(response['body'].read())
            output_text = response_body.get('completion', '')

            skills_list = []
            match = re.search(r"\[([^\]]+)\]", output_text)
            if match:
                raw_items = match.group(1).split(',')
                skills_list = [item.strip().strip("'\"").lower() for item in raw_items if item.strip()]

            logger.info(f"LLM Extracted Skills: {text}")
            return list(set(skills_list))  # remove duplicates

        except botocore.exceptions.ClientError as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"[Claude] Retry {attempt+1}/{retries}: waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
