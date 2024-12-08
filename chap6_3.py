import os
import random
from langchain.agents import AgentType, Tool, initialize_agent
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

tools = []

tools.append(WriteFileTool(
    root_dir="./"
))

def min_limit_random_number(min_number):
    return random.randint(int(min_number), 100000)

tools.append(
    Tool(
        name="Random",
        description="特定の最小値以上のランダムな数字を生成することができます。",
        func=min_limit_random_number
    )
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("10以上のランダムな数字を生成してrandom.txtというファイルに保存してください。")

print(f"実行結果:{result}")