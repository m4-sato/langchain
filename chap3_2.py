# from langchain.document_loaders import PyMuPDFLoader
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma  # ここでタイポ修正
from dotenv import load_dotenv

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load()

# 環境変数をロード
load_dotenv()


print(f"ドキュメントの数:{len(documents)}")
print(f"一つ目のドキュメント:{documents[0].page_content}")
print(f"一つ目のドキュメント:{documents[0].metadata}")

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ja_core_news_sm"
)

splitted_documents = text_splitter.split_documents(documents)

print(f"分割前:{len(documents)}")
print(f"分割跡:{len(splitted_documents)}")


embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    chunk_size=2048
    )


database = Chroma(
    persist_directory="./data",
    embedding_function=embeddings
)

database.add_documents(
    splitted_documents,
)

print('データベース登録完了')

#########################################################################
documents = database.similarity_search('飛行車の最高速度は？')

print(f"ドキュメントの数: {len(documents)}")

for document in documents:
    print(f"ドキュメントの内容: {document.page_content}")