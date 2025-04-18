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
    Calls Anthropic Claude v2 via Bedrock to extract details
    from a given resume or job description text.

    Returns:
        Dictionary: Extracted details, including Name, Email, Location,
                    Years of Experience, and Technical Skills.
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
Human: 
Extract the following details from the given resume text:
- Full Name
- Email address
- Location
- Years of professional experience
- Technical skills (include only programming languages, frameworks, libraries, software tools, cloud platforms, and technical certifications. Do not include soft skills, company names, locations, or general strengths)

Please return the result in a table format with the following columns:
| Name | Email | Location | Years of Experience | Technical Skills |

Rules:
1. For "Years of Experience":
- Identify the start and end dates for each role listed under the "Experience" section.
- If a role is marked as "Present," "Current," or "Till now," use today's date as the end date.
- Combine overlapping or continuous roles to avoid double-counting time periods.
- For overlapping periods, use the earliest start date and the latest end date within the overlap.
- Sum the total time span across all unique work periods.
- Convert the total time span into years, rounding down to the nearest whole number.

2. If any of the columns (Name, Email, Location, Years of Experience, Technical Skills) are not found in the text, write "Didn't Found" for that column.

Text:
\"\"\"{text}\"\"\"

Assistant:
"""
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

            line = output_text.strip().split('\n')[-1]
            parts = [p.strip() for p in line.split('|')]
            parts = parts[1:-1]

            name = ''
            email = ''
            location = ''
            years_of_experience = ''
            skills_list = []
            
            name = "Unable to Fetch" if parts[0] == '' or parts[0] == "Didn't Found" else parts[0]
            email = "Didn't Found" if parts[1] == '' else parts[1]
            location = "Didn't Found" if parts[2] == '' else parts[2]
            years_of_experience = "Didn't Found" if parts[3] == '' else parts[3]
            skills_match = "Didn't Found" if parts[4] == '' else parts[4]

            if skills_match:
                raw_items = skills_match.split(',')
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
            

def feedback_generation(resume_text, job_text, aws_access_key, aws_secret_key, aws_region, retries=3):
    """
    Calls Anthropic Claude v2 via Bedrock to generate feedback
    by comparing a given Resume to job description.

    Returns:
        String: Feedback highlighting the candidate's suitability based on the job description.
    """
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
Human: 
Based on the provided resume and job description, evaluate the candidate's suitability for the job.
Compare the content of resume with the job description, and provide constructive feedback. Follow the rules.

Rules:
    - Provide feedback of candidate in exactly 1 bullet point.
    - Provide suitability of candidate in exactly 1 bullet point.
    - Keep it precise and short.
    - Do not give long paragraph.
    - Give in one line.

Text: 
\"\"\"Resume : {resume_text}
Job Description : {job_text}\"\"\"

Assistant: 
"""
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 1000,
        "temperature": 0.2,
        "top_p": 0.5,
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
            feedback = []
            feedback = output_text.strip().split('\n')
            feedback = [p for p in feedback if p != '']
            feedback = "\n".join(feedback)

            return feedback
        
        except botocore.exceptions.ClientError as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"[Claude] Retry {attempt+1}/{retries}: waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e