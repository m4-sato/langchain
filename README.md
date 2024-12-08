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
- RAG実装手順
  - step1. 事前準備
    - 1-1.テキスト抽出(Document loaders)
    - 1-2.テキスト分割(Text splitters)
    - 1-3.テキストのベクトル化(Text embedding models)
    - 1-4.テキストとベクトルをベクトルデータベースに保存(Vector stores)
  - step2. 構築とプロンプト構築
    - 2-1. ユーザーからの入力をベクトル化(Text embedding models)
    - 2-2. ユーザー入力のベクトルを事前準備したデータベースで検索して、文章を取得する(Vector Stores)
    - 2-3. 取得した類似文章と質問を組み合わせてプロンプトを作成(PromptTemplate)
    - 2-4. 作成したプロンプトを使って言語モデルを呼び出す(Language models)
## Memory
- 過去の会話履歴情報を保持
## Chains
- 複数のモジュールを組み合わることを担う
## Agents
- ReActやOpenAI Function Callingという手法を使い、言語モデルの呼び出しでは対応できないタスクを実行する役割を担う
## Callbacks
- イベント発生時の処理の役割を担う

### 参考資料

- https://github.com/harukaxq/langchain-book
- https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIServicePracticalGuide-book
- [Flowise](https://flowiseai.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/jp)
-[awesome-langchain](https://github.com/kyrolabs/awesome-langchain)


### サンプルデータ
- https://raw.githubusercontent.com/harukaxq/langchain-book/master/asset/sample.pdf


### 調べる用プロンプト

# 役割
  優秀なPythonプログラマー

# 指示
  以下のソースコードを実行したところpydantic関係のエラーが出ました。

# 条件
- AzureOpenAISeriveiceAPI
- Pythonライブラリ
  - openai
  - python-dotenv
  - langchain
  - langchain_openai


# ソースコード
```python
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
        "serpapi"
    ]
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
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
```

# エラーコード
"""
(test-py3.10) C:\Users\mssst\Git\langchain-1>chainlit run chap7_1.py
2024-12-08 14:27:33 - Loaded .env file
C:\Users\mssst\Git\langchain-1\chap7_1.py:3: LangChainDeprecationWarning: Importing load_tools from langchain.agents is deprecated. Please replace deprecated imports:

>> from langchain.agents import load_tools

with new imports of:

>> from langchain_community.agent_toolkits.load_tools import load_tools
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/docs/versions/v0_2/>
  from langchain.agents import AgentType, initialize_agent, load_tools
C:\Users\mssst\Git\langchain-1\chap7_1.py:24: LangChainDeprecationWarning: The function `initialize_agent` was deprecated in LangChain 0.1.0 and will be removed in 1.0. Use :meth:`~Use new agent constructor methods like create_react_agent, create_json_agent, create_structured_chat_agent, etc.` instead.
  agent = initialize_agent(
2024-12-08 14:27:37 - Your app is available at http://localhost:8000
2024-12-08 14:27:38 - Translation file for ja not found. Using default translation en-US.
2024-12-08 14:27:38 - Translated markdown file for ja not found. Defaulting to chainlit.md.
C:\Users\mssst\Git\langchain-1\chap7_1.py:38: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
  result = agent.run(


> Entering new AgentExecutor chain...
2024-12-08 14:28:09 - HTTP Request: POST https://gpt4o-sample-test.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview "HTTP/1.1 200 OK"
2024-12-08 14:28:09 - An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: Could not parse LLM output: It looks like there was an error or an incomplete message. Could you please provide more details or clarify your question?
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE
Traceback (most recent call last):
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\chat\output_parser.py", line 47, in parse
    raise ValueError("action not found")
ValueError: action not found

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\agent.py", line 1363, in _iter_next_step
    output = self._action_agent.plan(
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\agent.py", line 810, in plan
    return self.output_parser.parse(full_output)
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\chat\output_parser.py", line 62, in parse
    raise OutputParserException(
langchain_core.exceptions.OutputParserException: Could not parse LLM output: It looks like there was an error or an incomplete message. Could you please provide more details or clarify your question?
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\chainlit\utils.py", line 44, in wrapper
    return await user_function(**params_values)
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\chainlit\callbacks.py", line 118, in with_parent_id
    await func(message)
  File "C:\Users\mssst\Git\langchain-1\chap7_1.py", line 38, in on_message
    result = agent.run(
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain_core\_api\deprecation.py", line 182, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\chains\base.py", line 606, in run
    return self(args[0], callbacks=callbacks, tags=tags, metadata=metadata)[
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain_core\_api\deprecation.py", line 182, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\chains\base.py", line 389, in __call__
    return self.invoke(
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\chains\base.py", line 170, in invoke
    raise e
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\chains\base.py", line 160, in invoke
    self._call(inputs, run_manager=run_manager)
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\agent.py", line 1629, in _call
    next_step_output = self._take_next_step(
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\agent.py", line 1335, in _take_next_step
    [
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\agent.py", line 1335, in <listcomp>
    [
  File "C:\Users\mssst\Git\langchain-1\.venv\lib\site-packages\langchain\agents\agent.py", line 1374, in _iter_next_step
    raise ValueError(
ValueError: An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: Could not parse LLM output: It looks like there was an error or an incomplete message. Could you please provide more details or clarify your question?
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE
"""