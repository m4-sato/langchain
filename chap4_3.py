import os
from typing import List
import chainlit as cl
from langchain.chains import ConversationChain
# from langchain.chat_models import AzureChatOpenAI
# from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 環境変数をロード
load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
    )

memory = ConversationBufferMemory(
    return_messages=True
)


chain = ConversationChain(
    memory=memory,
    llm=chat,
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="私は会話の履歴を考慮した返答ができるチャットボットです。メッセージを入力してください").send()

@cl.on_message
async def on_message(message):

    result = chain(message.content)

    await cl.Message(content=result["response"]).send()