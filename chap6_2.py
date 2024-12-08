import os
import chainlit as cl
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import AzureChatOpenAI
from langchain.tools.file_management import WriteFileTool
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

tools = load_tools(
    ["requests_get","serpapi"],
    allow_dangerous_tools=True,
    llm=chat
)

tools.append(WriteFileTool(
    root_dir="./"
))

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("北海道の名産品を調べて日本語でresult.txtというファイルに保存してください。")

print(f"実行結果:{result}")