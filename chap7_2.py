import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()


class LogCallbackHandler(BaseCallbackHandler):
    
    def on_chat_model_start(self, serialized, messages, **kwards):
        print("ChatModelの実行を開始します...")
        print(f"入力:{messages}")
    
    def on_chain_start(self, serialized, inputs, **kwards):
        print("Chainの実行を開始します....")
        print(f"入力：{inputs}")

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
    callbacks=[
        LogCallbackHandler()
    ]
)

result = chat([
    HumanMessage(content="こんにちは!"),
])

print(result.content)