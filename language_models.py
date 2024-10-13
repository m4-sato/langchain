import os
import openai
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# Azure OpenAI APIの設定
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY =  os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME =  os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # ここはご自身のデプロイメント名に置き換えてください

chat = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2024-07-01-preview",
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_type="azure"
)

output = chat(
    [
        HumanMessage(content="茶碗蒸しの作り方を教えて"),
        AIMessage(content="{ChatModelからの返答である茶碗蒸しの作り方}"),
        HumanMessage(content="英語に翻訳して")
    ]
)

print(output.content)
