import os
import chainlit as cl
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    chunk_size=2048
    )

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
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






# @cl.on_message
# async def on_message(input_message):
#     print("入力されたメッセージ:" + input_message.content)

#     database = cl.user_session.get("database")

#     documents = database.similarity_search(input_message.content)

#     documents_string = ""

#     for document in documents:
#         documents_string += f"""
#     -----------------------------
#     {document.page_content}
#     """

#     result = llm([
#         HumanMessage(content=prompt.format(document=documents_string, query=input_message.content))
#     ])

#     await cl.Message(content=result.content).send()