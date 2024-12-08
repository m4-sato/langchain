import os
import chainlit as cl
from langchain.chains import LLMChain, LLMRequestsChain
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
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
    input_variables=[
        "query",
        "requests_result"
        ],
    template="""以下の文章を元に質問に答えてください。
    文章:{requests_result}
    質問:{query}"""
)

llm_chain = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True
)

chain = LLMRequestsChain(
    llm_chain=llm_chain
)

print(chain({
    "query":"東京の天気について教えて",
    "url":"https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json"
}))