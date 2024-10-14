import chainlit as cl
import os
import openai
from langchain.chat_models import AzureChatOpenAI  # 修正箇所
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate  # ChatPromptTemplate -> PromptTemplateに修正
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage  # 修正箇所
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# Azure OpenAI APIの設定
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY =  os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME =  os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # ここはご自身のデプロイメント名に置き換えてください

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    chunk_size=2048  # chunk_sizeを明示的に設定
)

chat = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2024-07-01-preview",
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_type="azure"
)

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。
    
文章:
{document}

質問: {query}
""",
    input_variables=["document", "query"]
)

database = Chroma(
    persist_directory="./data",
    embedding_function=embeddings
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="準備ができました！メッセージを入力してください！").send()

@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ:" + input_message.content)
    documents = database.similarity_search(input_message.content)
    await cl.Message(content="こんにちは！").send()

    documents_string = ""

    for document in documents:
        documents_string += f"""
    -----------------------------
    {document.page_content}
    """

    result = chat([
        HumanMessage(content=prompt.format(document=documents_string, query=input_message.content))
    ])
    await cl.Message(content=result.content).send()