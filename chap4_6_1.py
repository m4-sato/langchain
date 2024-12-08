import os
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# ... (前述のDatabaseManagerとChatbotManagerクラスは同じ) ...

async def get_valid_user_input(prompt: str, 
                             validator: callable = None, 
                             error_message: str = None,
                             max_attempts: int = 3) -> Tuple[bool, Optional[str]]:
    """
    ユーザー入力を取得し、バリデーションを行う
    
    Args:
        prompt: 表示するプロンプト
        validator: バリデーション関数
        error_message: エラー時のメッセージ
        max_attempts: 最大試行回数
    
    Returns:
        (成功したかどうか, 入力値)のタプル
    """
    attempts = 0
    while attempts < max_attempts:
        try:
            response = await cl.AskUserMessage(content=prompt, timeout=120).send()
            if not response:
                await cl.Message(content="タイムアウトしました。もう一度お試しください。").send()
                attempts += 1
                continue

            user_input = response.get('content', '').strip()
            
            if not user_input:
                if error_message:
                    await cl.Message(content=error_message).send()
                attempts += 1
                continue

            if validator and not validator(user_input):
                if error_message:
                    await cl.Message(content=error_message).send()
                attempts += 1
                continue

            return True, user_input

        except Exception as e:
            await cl.Message(content=f"入力エラーが発生しました: {str(e)}").send()
            attempts += 1

    return False, None

def validate_user_id(user_id: str) -> bool:
    """ユーザーIDのバリデーション"""
    return len(user_id) >= 3 and len(user_id) <= 50

@cl.on_chat_start
async def on_chat_start():
    try:
        chatbot = ChatbotManager()
        cl.user_session.set("chatbot", chatbot)

        # データベースの初期化
        await chatbot.db_manager.initialize_database()

        # ユーザーIDの取得
        success, user_id = await get_valid_user_input(
            prompt="ユーザーIDを入力してください（3-50文字）",
            validator=validate_user_id,
            error_message="ユーザーIDは3文字以上50文字以下で入力してください。",
            max_attempts=5
        )

        if not success or not user_id:
            await cl.Message(content="ユーザーIDの入力に失敗しました。チャットを終了します。").send()
            return

        # ユーザーの会話一覧を表示
        conversations = await chatbot.db_manager.get_user_conversations(user_id)
        
        if conversations:
            conversation_list = "\n".join([
                f"ID: {conv['conversation_id']} - {conv['title']} "
                f"({conv['created_at'].strftime('%Y-%m-%d %H:%M:%S')})"
                for conv in conversations
            ])
            await cl.Message(content=f"既存の会話:\n{conversation_list}").send()
            await cl.Message(content="\n新しい会話を作成する場合は 'new' と入力してください。").send()
        else:
            await cl.Message(content="会話履歴がありません。新しい会話を作成します。").send()
            conversations = []

        # 新しい会話を作成するか既存の会話を選択
        success, action = await get_valid_user_input(
            prompt="新しい会話を作成する場合は'new'、既存の会話を選択する場合は会話IDを入力してください",
            error_message="無効な入力です。もう一度入力してください。",
            max_attempts=3
        )

        if not success:
            await cl.Message(content="会話の選択に失敗しました。チャットを終了します。").send()
            return

        conversation_id = None
        if action.lower() == 'new':
            success, title = await get_valid_user_input(
                prompt="新しい会話のタイトルを入力してください（1-200文字）",
                validator=lambda x: 1 <= len(x) <= 200,
                error_message="タイトルは1文字以上200文字以下で入力してください。",
                max_attempts=3
            )

            if not success:
                await cl.Message(content="タイトルの入力に失敗しました。チャットを終了します。").send()
                return

            conversation_id = await chatbot.db_manager.create_conversation(
                user_id, title
            )
            await cl.Message(content=f"新しい会話を作成しました: {conversation_id}").send()
        else:
            # 既存の会話IDのバリデーション
            valid_ids = [conv['conversation_id'] for conv in conversations]
            if action not in valid_ids:
                await cl.Message(content="無効な会話IDです。チャットを終了します。").send()
                return
            conversation_id = action

        # 会話チェーンの作成
        chain = await chatbot.create_conversation_chain(conversation_id)
        cl.user_session.set("chain", chain)
        cl.user_session.set("conversation_id", conversation_id)
        cl.user_session.set("user_id", user_id)
        
        await cl.Message(content=f"会話ID: {conversation_id} で開始します。\nメッセージを入力してください。").send()

    except Exception as e:
        await cl.Message(content=f"初期化エラー: {str(e)}\nチャットを終了します。").send()

# ... (残りのコードは同じ) ...