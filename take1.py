import os
import chainlit as cl
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, Document
from dotenv import load_dotenv
import shutil
from pydantic import BaseModel
from typing import List

load_dotenv()

# Embeddings
embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    chunk_size=2048
)

# LLM
chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-08-01-preview",
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# プロンプトテンプレート
prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。

文章:
{context}

質問: {question}
""",
    input_variables=["context", "question"]
)

# テキスト分割
text_splitter = SpacyTextSplitter(
    chunk_size=1000,
    pipeline="ja_core_news_sm"
)


@cl.on_chat_start
async def on_chat_start():
    """初回起動時にファイルのアップロードを促し、Chroma に取り込み、会話チェーンをセットする。"""

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="PDFを選択してください",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()

    # 1つ目のファイルを取得
    file = files[0]
    print("file:", file)
    print("type(file):", type(file))

    # 一時フォルダがなければ作成
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    # PDF を一時フォルダにコピー
    shutil.copy(file.path, f"tmp/{file.name}")
    documents = PyMuPDFLoader(file.path).load()

    # テキスト分割
    splitted_documents = text_splitter.split_documents(documents)

    # データベースを初期化し、Embedding したドキュメントを格納
    database = Chroma(
        embedding_function=embeddings,
    )
    
    database.add_documents(splitted_documents)
    cl.user_session.set("database", database)

    # docsearch: デモ用に async で作ってみた例
    docsearch = await cl.make_async(Chroma.from_documents)(
        documents, embeddings
    )

    # チャットメモリ
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # ConversationalRetrievalChain の初期化
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt  # カスタムプロンプト
        },
    )

    # 過去メッセージの取得（同期メソッドは削除されたため非同期版）
    messages = await chain.memory.chat_memory.aget_messages()

    # 過去メッセージがあれば画面に表示
    for message in messages:
        if isinstance(message, HumanMessage):
            await cl.Message(
                author="User",
                content=message.content,
            ).send()
        elif isinstance(message, AIMessage):
            await cl.Message(
                author="ChatBot",
                content=message.content,
            ).send()

    # セッションに chain を保存
    cl.user_session.set("chain", chain)

    await cl.Message(
        content=f"`{file.name}`の読み込みが完了しました。質問を入力してください。"
    ).send()


@cl.on_message
async def on_message(input_message: cl.Message):
    """ユーザーからの質問を受け取り、LLM に問い合わせ、回答と参照元を右カラムのサイドバーに表示する。"""

    chain: ConversationalRetrievalChain = cl.user_session.get("chain")

    if chain is None:
        await cl.Message(content="Chain が初期化されていません。PDF をアップロードし直してください。").send()
        return

    print("入力されたメッセージ:", input_message.content)

    # LLM に問い合わせ
    result = chain({"question": input_message.content})

    # 結果には answer と source_documents が含まれる
    answer = result["answer"]
    source_docs = result["source_documents"]

    # サイドバーに表示する要素を作成
    elements = []
    if source_docs:
        for i, doc in enumerate(source_docs):
            page_num = doc.metadata.get("page", "N/A")
            # テキストが長い場合は適当にトリム
            snippet = doc.page_content[:200].replace("\n", " ")
            if len(doc.page_content) > 200:
                snippet += "..."

            element_content = f"**Source {i+1}** (page: {page_num})\n{snippet}"
            element = cl.Text(
                name=f"source_{i+1}",  # サイドバーの折りたたみ名
                content=element_content
            )
            elements.append(element)

    else:
        # ソースがない場合に何か表示したいなら
        element = cl.Text(
            name="source_0",
            content="(参照元の情報はありませんでした)"
        )
        elements.append(element)

    # まとめてメッセージ送信。elements により、右カラムにソース一覧が表示される
    await cl.Message(
        content=answer,  # 左カラムに表示される回答
        elements=elements
    ).send()
