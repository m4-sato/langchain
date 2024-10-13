import os
import openai
from langchain_openai import AzureChatOpenAI
from langchain import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from pydantic import field_validator
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

class Smartphone(BaseModel):
    release_date: str = Field(description='スマートフォンの発売日')
    screen_inches: float = Field(description='スマートフォンの画面サイズ（インチ）')
    os_installed: str = Field(description='スマートフォンにインストールされているOS')
    model_name: str = Field(description='スマートフォンのモデル名')
    
    @field_validator("screen_inches")
    def validate_screen_inches(cls, value):
        if value <= 0:
            raise ValueError("Screen inches must be a positive number")
        return value

parser = OutputFixingParser.from_llm(
    parser = PydanticOutputParser(pydantic_object=Smartphone),
    llm=chat
)

result = chat([
    HumanMessage(content="Androidでリリースしたスマートフォンを1個あげて"),
    HumanMessage(content=parser.get_format_instructions())
])

parsed_result = parser.parse(result.content)

print(f"モデル名：{parsed_result.model_name}")
print(f"画面サイズ: {parsed_result.screen_inches}インチ")
print(f"OS: {parsed_result.os_installed}")
print(f"スマートフォンの発売日: {parsed_result.release_date}")