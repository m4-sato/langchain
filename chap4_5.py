import os
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
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


# データベース接続情報を環境変数から取得
postgres_user = os.getenv("POSTGRES_USER", "postgres")
postgres_password = os.getenv("POSTGRES_PASSWORD")
postgres_host = os.getenv("POSTGRES_HOST", "localhost")
postgres_port = os.getenv("POSTGRES_PORT", "5432")
postgres_db = os.getenv("POSTGRES_DB", "chat_history")


# 接続文字列の構築
connection_string = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

history = PostgresChatMessageHistory(
    connection_string=connection_string,
    session_id="chat_history",
)

memory = ConversationBufferMemory(
    return_messages=True,
    chat_memory=history,
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
    # user_message = message.content

    # チェーンを実行
    # result = chain.run(input=user_message)
    result = chain(message.content)

    await cl.Message(content=result["response"]).send()