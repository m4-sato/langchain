import os
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import AzureChatOpenAI
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.schema import messages_from_dict
from dotenv import load_dotenv
import uuid
import asyncpg
import psycopg  # psycopgを使用
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
# from langchain.memory.chat_memory import ChatMessageHistory


# 環境変数をロード
load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)



# Establish a synchronous connection to the database
# (or use psycopg.AsyncConnection for async)
conn_info = "dbname=chat_history user=postgres password=password host=localhost port=5432"# Fill in with your connection info
sync_connection = psycopg.connect(conn_info)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

session_id = str(uuid.uuid4())

# @cl.on_chat_start
# async def on_chat_start():
#     thread_id = None
#     while not thread_id:
#         res = await cl.AskUserMessage(
#             content="私は会話の履歴を考慮した返答ができるチャットボットです。スレッドIDを入力してください（例: 123e4567-e89b-12d3-a456-426614174000）", timeout=600
#         ).send()
#         if res:
#             input_thread_id = res.get("output", "")
#             try:
#                 uuid.UUID(input_thread_id)  # UUID形式を検証
#                 thread_id = input_thread_id
#             except ValueError:
#                 await cl.Message(content="無効なスレッドIDが入力されました。新しいスレッドIDを生成します。").send()
#                 thread_id = str(uuid.uuid4())

#     # 非同期接続を初期化
#     async_connection = await asyncpg.connect(
#         user=os.getenv("POSTGRES_USER", "postgres"),
#         password=os.getenv("POSTGRES_PASSWORD"),
#         database=os.getenv("POSTGRES_DB", "chat_history"),
#         host=os.getenv("POSTGRES_HOST", "localhost"),
#         port=os.getenv("POSTGRES_PORT", "5432"),
#     )

#     # PostgresChatMessageHistoryを初期化
#     history = PostgresChatMessageHistory(
#         table_name,
#         session_id,
#         sync_connection=sync_connection
#         )

#     # 会話メモリを設定
#     memory = ConversationBufferMemory(
#         return_messages=True,
#         chat_memory=history,
#     )

#     # チェーンを作成
#     chain = ConversationChain(
#         memory=memory,
#         llm=chat,
#     )

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def on_message(message: str):
#     chain = cl.user_session.get("chain")
#     user_input = str(message)
#     result = await chain.ainvoke({"input": user_input})  # 修正済み: ainvokeを使用
#     await cl.Message(content=result["response"]).send()


@cl.on_chat_start
async def on_chat_start():
    thread_id = None
    while not thread_id:
        res = await cl.AskUserMessage(
            content="私は会話の履歴を考慮した返答ができるチャットボットです。スレッドIDを入力してください（例: 123e4567-e89b-12d3-a456-426614174000）", timeout=600
        ).send()
        if res:
            input_thread_id = res.get("output", "")
            try:
                uuid.UUID(input_thread_id)  # UUID形式を検証
                thread_id = input_thread_id
            except ValueError:
                await cl.Message(content="無効なスレッドIDが入力されました。新しいスレッドIDを生成します。").send()
                thread_id = str(uuid.uuid4())

    # 非同期接続を初期化
    async_connection = await asyncpg.connect(
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB", "chat_history"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )

    # PostgresChatMessageHistoryを初期化
    history = PostgresChatMessageHistory(
        table_name=table_name,
        session_id=thread_id,
        async_connection=async_connection,
    )

    # ChatMessageHistory を初期化
    chat_memory = ChatMessageHistory(
        chat_message_history=history
    )

    # RunnableWithMessageHistory を使用してチェーンを作成
    chain = RunnableWithMessageHistory(
        history=chat_memory,
        runnable=chat,  # Azure OpenAI モデルを使用
    )

    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: str):
    chain = cl.user_session.get("chain")
    user_input = str(message)

    # 新しいチェーンの呼び出し方法に変更
    result = await chain.arun(user_input)

    # AIからの返答をユーザーに送信
    await cl.Message(content=result).send()
