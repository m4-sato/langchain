import os
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
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
    openai_api_type="azure",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)
resp = chat([
    HumanMessage(content="おいしいステーキの焼き方を教えて")
])

response_text = resp.content