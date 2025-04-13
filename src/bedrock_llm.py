import boto3
import json
import re
import time
import botocore.exceptions
from src.logger import logger
#from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
import os

def set_bedrock_credentials(access_key, secret_key, region):
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["AWS_REGION"] = region


def run_skill_extraction_prompt(text, aws_access_key, aws_secret_key, aws_region, retries=3):
    """
    Calls Anthropic Claude v2 via Bedrock to extract technical skills
    from a given resume or job description text.

    Returns:
        List[str]: Cleaned list of skill strings (lowercase, no duplicates).
    """
    #client = boto3.client(service_name='bedrock-runtime')

    # client = boto3.client(
    #     service_name='bedrock-runtime',
    #     aws_access_key_id=AWS_ACCESS_KEY,
    #     aws_secret_access_key=AWS_SECRET_KEY,
    #     region_name=AWS_REGION
    # )
    client = boto3.client(
        service_name='bedrock-runtime',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

    model_id = 'anthropic.claude-v2'
    accept = 'application/json'
    content_type = 'application/json'

    prompt = f"""
Human: Extract the following details from the given resume text:
Full Name, Email address, Location, Years of professional experience, Technical skills (programming languages, frameworks, libraries, software tools, cloud platforms, and technical certifications only; exclude soft skills, company names, locations, and general strengths)
Return the result in a table format with the following columns:
Name | Email | Location | Years of Experience | Technical Skills
Ensure technical skills are returned as a valid Python list of strings in the "Technical Skills" column.


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

            name = ''
            email = ''
            location = ''
            years_of_experience = ''
            skills_list = []
            parts = []

            lines = output_text.strip().split('\n')
            for line in lines:
                if '|' in line and '@' in line:  # likely the data line
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 5:
                        name = parts[1]
                        email = parts[2]
                        location = parts[3]
                        years_of_experience = parts[4]
                    break


            # Extract Skills 
            skills_match = re.search(r"\[([^\]]+)\]", output_text)
            if skills_match:
                raw_items = skills_match.group(1).split(',')
                skills_list = [item.strip().strip("'\"").lower() for item in raw_items if item.strip()]
                skills_list = list(set(skills_list))

            logger.info(f"LLM Extracted: Name : {name}, Email : {email}, Location : {location} Years of Experience : {years_of_experience}, Skills : {skills_list}")
            return {
                "Name": name,
                "Email": email,
                "Location": location,
                "Years of Experience": years_of_experience,
                "Technical Skills": skills_list
            }

        except botocore.exceptions.ClientError as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"[Claude] Retry {attempt+1}/{retries}: waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
