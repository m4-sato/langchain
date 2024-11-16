import os
# from langchain.chat_models import AzureChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
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

database = Chroma(
    persist_directory="./data",
    embedding_function=embeddings
)


# query = '飛行車の最高速度は？'
query = '飛行車の違反速度時の罰金は？'

documents = database.similarity_search(query)

documents_string = ""
print(documents)

for document in documents:
    print(document)
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
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

result = chat([
    HumanMessage(content=prompt.format(document=documents_string, query=query))
])

print(result.content)