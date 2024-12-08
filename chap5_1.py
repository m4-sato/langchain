import os
import chainlit as cl
from langchain import LLMChain, PromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
    )

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？",
    input_variables=[
        "product"
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True
)

result = chain.predict(product="iPhone")

print(result)