import os
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

result = chat(
    [
        HumanMessage(content="茶碗蒸しを作るにはどんな材料が必要ですか？"),
        AIMessage(
            content="""茶碗蒸しを作るためには、以下のような材料が必要です：
            1. 卵 - 2個
            2. だし汁 - 約300ml（昆布だしや鰹だしを使うのが一般的です）
            3. 醤油 - 小さじ1
            4. みりん - 小さじ1
            5. 塩 - 少々
            これらの材料を使って茶碗蒸しを作ることができます。具体的なレシピに従って手順を進めると良いでしょう。"""),
        HumanMessage(content="前の回答を英語に翻訳して")
        ])

print(result.content)