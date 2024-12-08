import os
import chainlit as cl
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
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

write_article_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate(
        template="{input}についての記事を書いてください。",
        input_variables=["input"],
    )
)


translate_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate(
        template="以下の文章を英語に翻訳してください。\n{input}",
        input_variables=["input"],
    ),
)


sequential_chain = SimpleSequentialChain(
    chains=[
        write_article_chain,
        translate_chain,
    ]
)

result = sequential_chain.run("エレキギターの選び方")

print(result)