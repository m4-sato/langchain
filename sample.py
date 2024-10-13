import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import requests

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.chat.completions.create(
    model="gpt-4o", 
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Who were the founders of Microsoft?"}
    ],
    max_tokens = 100,
    temperature = 1,
    n = 1,
)

# print(json.dumps(response, indent=2,))
print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)