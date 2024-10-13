import os
import openai
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

output_parser = CommaSeparatedListOutputParser()

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

# # テンプレートの作成
# prompt = ChatPromptTemplate.from_template(
#     template="{product}はどこの会社が開発した製品ですか？"
# )

# # フォーマットと出力
# print(prompt.format(product="iPhone"))  # 出力: iPhoneはどこの会社が開発した製品ですか？


# result = chat(
#     [
#     HumanMessage(content=prompt.format(product="iPhone")),
#     ]
# )

result = chat(
    [
    HumanMessage(content="Appleが開発した代表的な製品を3つ教えてください"),
    HumanMessage(content=output_parser.get_format_instructions()),
    ]
)

output = output_parser.parse(result.content)

for item in output:
    print('代表的な製品 =>' + item)