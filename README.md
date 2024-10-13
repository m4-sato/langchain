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
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

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

# テンプレートの作成
prompt = ChatPromptTemplate.from_template(
    template="{product}はどこの会社が開発した製品ですか？"
)

# フォーマットと出力
print(prompt.format(product="iPhone"))  # 出力: iPhoneはどこの会社が開発した製品ですか？

result = chat(
    [
    HumanMessage(content=prompt.format(product="iPhone")),
    ]
)

print(result.content)

prompt_json = prompt.save("prompt.json")
```

# エラーコード
"""
(venv) C:\Users\mssst\Git\langchain-1>python prompt_langage_model.py
Human: iPhoneはどこの会社が開発した製品ですか？
C:\Users\mssst\Git\langchain-1\prompt_langage_model.py:32: LangChainDeprecationWarning: The method `BaseChatModel.__call__` was deprecated in langchain-core 0.1.7 and will be removed in 1.0. Use :meth:`~invoke` instead.
  result = chat(
iPhoneは、アメリカのApple Inc.（アップル社）が開発した製品です。Appleは2007年に初代iPhoneを発表し、それ以来、毎年新しい モデルをリリースしています。
Traceback (most recent call last):
  File "C:\Users\mssst\Git\langchain-1\prompt_langage_model.py", line 40, in <module>
    prompt_json = prompt.save("prompt.json")
  File "C:\Users\mssst\Git\langchain-1\venv\lib\site-packages\langchain_core\prompts\chat.py", line 1339, in save
    raise NotImplementedError
NotImplementedError
"""