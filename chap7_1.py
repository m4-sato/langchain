import os
import chainlit as cl
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import AzureChatOpenAI
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
    [
        "serpapi",
    ]
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Agentの初期化が完了しました").send()

@cl.on_message
async def on_message(input_message):
    result = agent.run(
        input_message,
        callbacks = [
            cl.LangchainCallbackHandler()
        ]
    )
    await cl.Message(content=result).send()