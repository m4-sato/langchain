# langchain

## 環境構築
- VSCodeのインストール
  - VSCode拡張機能「Pylance」のインストール（Pythonのソースコード解析・入力補完機能）

# Langchainの主要モジュール

## Model I/O
- LLMを呼び出すための「プロンプトの準備」、「言語モデルの呼び出し」、「結果の受け取り」という3つのステップの担う
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
- 