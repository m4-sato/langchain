import os
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma  # ここでタイポ修正

# 環境変数をロード
load_dotenv()

# Azure OpenAI APIの設定
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY =  os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME =  os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # ここはご自身のデプロイメント名に置き換えてください


pdf_loader = PyPDFLoader("./sample.pdf")
documents = pdf_loader.load()

# テキスト分割器の設定（ここでは1000文字ごとに分割）
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 分割されたテキストリストを生成
split_texts = text_splitter.split_documents(documents)


embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    chunk_size=2048  # chunk_sizeを明示的に設定
)

database = Chroma(
    persist_directory="./data",
    embedding_function=embeddings
)

database.add_documents(
    split_texts,
)

print('データベース登録完了')