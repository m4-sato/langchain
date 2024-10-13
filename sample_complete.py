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

response = client.completions.create(
    engine="gpt-4o", 
    prompt="今日の天気がとてもよくて、気分が",
    stop="。",
    max_tokens = 100,
    n = 2,
    temperature = 0.5
)

# print(json.dumps(response, indent=2,))
print(response.model_dump_json(indent=2))