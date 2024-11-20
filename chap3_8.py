import os
from langchain.chat_models import AzureChatOpenAI
# from langchain.retrievers import WikipediaRetriever
from langchain_community.retrievers import WikipediaRetriever
from langchain.retrievers import RePhraseQueryRetriever
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv


# 環境変数をロード
load_dotenv()

prompt = PromptTemplate(
    input_variables=["question"],
    template="""以下の質問からWikipediaで検索するべきキーワードを抽出してください。
    質問：{question}
    """
)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
    )

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500
    )

# llm_chain = LLMChain(
#     llm = AzureChatOpenAI(
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_version="2024-08-01-preview",
#         model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         temperature=0
#         ),
#     prompt = PromptTemplate(
#         input_variables=["question"],
#         template="""以下の質問からWikipediaで検索するべきキーワードを抽出してください。
#         質問：{question}
#         """
# ))

# RunnableSequenceを作成
chain = prompt | llm

re_phrase_query_retriever = RePhraseQueryRetriever(
    llm_chain = chain,
    retriever = retriever,
    )

# documents = re_phrase_query_retriever.get_relevant_documents("私はラーメンが好きです。ところでバーボンウイスキーとは何ですか？")
# documents = re_phrase_query_retriever.invoke("質問文")
# print(documents)

# 質問を定義
question = "私はラーメンが好きです。ところでバーボンウイスキーとは何ですか？"

# RePhraseQueryRetrieverを使用して再検索クエリを取得
rephrased_query = re_phrase_query_retriever.llm_chain.invoke({"question": question})

# AIMessageからコンテンツを抽出
if hasattr(rephrased_query, 'content'):
    rephrased_query = rephrased_query.content

# 再検索クエリを使用してドキュメントを取得
documents = retriever.get_relevant_documents(rephrased_query)

print(documents)