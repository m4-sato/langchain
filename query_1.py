import os
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
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

database = Chroma(
    persist_directory="./data",
    embedding_function=embeddings
)

query = database.similarity_search('飛行車の最高速度は？')

print(f"ドキュメントの数: {len(documents)}")

for document in documents:
    print(f"ドキュメントの内容: {document.page_content}")