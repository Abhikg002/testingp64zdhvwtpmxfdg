import os
import json
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load credentials from .env if present
load_dotenv()

def test_titan_text_express():
    try:
        # Create Bedrock Runtime client
        bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

        model_id = "amazon.titan-text-express-v1"

        prompt = "Write a short paragraph about the benefits of exercise."

        # Titan model input format
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 200,
                "temperature": 0.5,
                "topP": 0.9,
                "stopSequences": []
            }
        })

        response = bedrock.invoke_model(
            modelId=model_id,
            body=body,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response["body"].read())
        output_text = response_body["results"][0]["outputText"]

        print("\n✅ LLM Response:")
        print(output_text)

    except ClientError as e:
        print("❌ AWS Client Error:", e.response["Error"]["Message"])
    except Exception as e:
        print("❌ General Error:", str(e))


if __name__ == "__main__":
    test_titan_text_express()
