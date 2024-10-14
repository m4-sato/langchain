import os
from langchain_openai import AzureOpenAIEmbeddings
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# Azure OpenAI APIの設定
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY =  os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME =  os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # ここはご自身のデプロイメント名に置き換えてください


embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    chunk_size=2048  # chunk_sizeを明示的に設定
)

query_vector = embeddings.embed_query("飛行車の最高速度は?")

print(f"ベクトル化された質問：{query_vector[:5]}")

document_1_vector = embeddings.embed_query("飛行車の最高速度は150キロメートルです。")
document_2_vector = embeddings.embed_query("鶏肉を適切に下味をつけた後、中火で焼きながらたまに裏返し、外側は香ばしく中は柔らかく仕上げる。")

cos_sim_1 = dot(query_vector, document_1_vector) / (norm(query_vector)*norm(document_1_vector))
print(f"ドキュメント1と質問の類似度: {cos_sim_1}")

cos_sim_2 = dot(query_vector, document_2_vector) / (norm(query_vector)*norm(document_2_vector))
print(f"ドキュメント1と質問の類似度: {cos_sim_2}")