import os
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_openai import AzureChatOpenAI
from langchain.retrievers import WikipediaRetriever
from langchain.tools import WriteFileTool
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
    )

tools = []

tools.append(WriteFileTool(
    root_dir="./"
))

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max = 500,
    top_k_results = 1
)

tools.append(
    create_retriever_tool(
        name="WikipediaRetreiver",
        description="受け取った単語に関するWikipediaの記事を取得できる",
        retriever=retriever
    )
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("スコッチウィスキーについてWikipediaで調べて概要を日本語でsample.txtというファイルに保存してください。")

print(f"実行結果:{result}")