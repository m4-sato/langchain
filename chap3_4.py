import chainlit as cl
import os
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage
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

# テキスト分割器の設定（ここでは1000文字ごとに分割）
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files= await cl.AskFileMessage(
            max_size_mb = 20,
            content="PDFを選択してください",
            accept = ["application/pdf"],
            raise_on_timeout=False,
        ).send()
    file = files.content

    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    with open(f"tmp/{files[0]['name']}", "wb") as f:
        f.write(file)
 
    documents = PyPDFLoader(f"tmp/{files[0]['name']}").load()
    
    # 分割されたテキストリストを生成
    split_texts = text_splitter.split_documents(documents)

    database = Chroma(
        embedding_function=embeddings,
    )

    database.add_documents(split_texts)

    cl.user_session.set(
        "database",
        database
    )


    await cl.Message(content=f"`{file.name}`の読み込みが完了しました。質問を入力してください。").send()


@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ:" + input_message.content)

    database = cl.user_session.get("database")

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