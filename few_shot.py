import os
import time
import langchain
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# Azure OpenAI APIの設定
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY =  os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME =  os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # ここはご自身のデプロイメント名に置き換えてください

chat = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2024-07-01-preview",
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_type="azure"
)


examples = [
    {
        "input": "LangChainはChatGPT・Large Language Model (LLM)の実利用をより柔軟に簡易に行うためのツール群です",  #← 入力例
        "output": "LangChainは、ChatGPT・Large Language Model (LLM)の実利用をより柔軟に、簡易に行うためのツール群です。"  #← 出力例
    }
]

prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="入力: {input}\n出力: {output}",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="以下の句読点の抜けた入力に句読点を追加してください。追加して良い句読点は「、」「。」のみです。他の句読点は追加しないでください。",
    suffix="入力: {input_string}\n出力:",
    input_variables=["input_string"],
)

formatted_prompt = few_shot_prompt.format(
    input_string="私はさまざまな機能がモジュールとして提供されているLangchainを使ってアプリケーションを開発しています"
)

result = chat.predict(formatted_prompt)
print("formatted_prompt", formatted_prompt)
print("result:", result)