# import os
# import openai
# from langchain_openai import AzureChatOpenAI
# from langchain_community.embeddings import AzureOpenAIEmbeddings
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain.prompts import ChatPromptTemplate
# from langchain.vectorstores import Chroma
# from dotenv import load_dotenv


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

database = Chroma(
    persist_directory="./data",
    embedding_function=embeddings
)

query = '飛行車の最高速度は？'

documents = database.similarity_search(query)

documents_string = ""

for document in documents:
    documents_string += f"""
-----------------------------
{document.page_content}
"""

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。
    
文章:
{document}

質問: {query}
""",
    input_variables=["document", "query"]
)

chat = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2024-07-01-preview",
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_type="azure"
)

result = chat([
    HumanMessage(content=prompt.format(document=documents_string, query=query))
])

print(result.content)