# langchain

## 環境構築
- VSCodeのインストール
  - VSCode拡張機能「Pylance」のインストール（Pythonのソースコード解析・入力補完機能）

# Langchainの主要モジュール

## Model I/O
- LLMを呼び出すための「プロンプトの準備」、「言語モデルの呼び出し」、「結果の受け取り」という3つのステップの担う
  - サブモジュール
    - 1. Language models
        様々な言語モデルを同一インターフェースで呼び出すための機能
    - 2. Prompts
        プロンプト構築の役割を担う
    - 3. Output parsers
        出力された結果を解析し、アプリで利用しやすい形へ変換
## Retrieval
- LLMが保持していない情報を扱う役割を担う
## Memory
- 過去の会話履歴情報を保持
## Chains
- 複数のモジュールを組み合わることを担う
## Agents
- ReActやOpenAI Function Callingという手法を使い、言語モデルの呼び出しでは対応できないタスクを実行する役割を担う
## Callbacks
- イベント発生時の処理の役割を担う

## 参考資料

- https://github.com/harukaxq/langchain-book
- https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIServicePracticalGuide-book
- [Flowise](https://flowiseai.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/jp)



### 調べる用プロンプト

# 役割
  優秀なPythonプログラマー

# 指示
  LangchainとAzureOpenAIServiceAPIを活用してLangchainのModel I/Oを検証したいです。

# 条件
- AzureOpenAISeriveiceAPI
- Pythonライブラリ
  - openai
  - python-dotenv
  - langchain
  - langchain_openai

# 目標
  LangchainのModel I/Oのソースコードで実行できる。


---
以下のソースコードを実行したところ次のエラーが出ました。

# ソースコード
```python
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

parser = OutputFixingParser.from_chat(
    parser = PydanticOutputParser(pydantic_object=Smartphone)
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
```

# エラーコード
"""
(venv) C:\Users\mssst\Git\langchain-1>python pydantic_output.py
C:\Users\mssst\Git\langchain-1\venv\lib\site-packages\langchain\__init__.py:30: UserWarning: Importing PromptTemplate from langchain root module is no longer supported. Please use langchain_core.prompts.PromptTemplate instead.
  warnings.warn(
C:\Users\mssst\Git\langchain-1\venv\lib\site-packages\pydantic\_internal\_fields.py:132: UserWarning: Field "model_name" in Smartphone has conflict with protected namespace "model_".

You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
  warnings.warn(
Traceback (most recent call last):
  File "C:\Users\mssst\Git\langchain-1\pydantic_output.py", line 41, in <module>
    parser = OutputFixingParser.from_chat(
  File "C:\Users\mssst\Git\langchain-1\venv\lib\site-packages\pydantic\_internal\_model_construction.py", line 262, in __getattr__
    raise AttributeError(item)
AttributeError: from_chat
"""