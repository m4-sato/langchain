import os
# from langchain.embeddings import AzureOpenAIEmbeddings
# from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv



# 環境変数をロード
load_dotenv()




embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    chunk_size=2048
    )

print("Deployment Name:", os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"))
print("Endpoint:", os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"))
print("API Key:", os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"))
print("API Version:", os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"))


query_vector = embeddings.embed_query("飛行車の最高速度は?")

print(f"ベクトル化された質問：{query_vector[:5]}")

document_1_vector = embeddings.embed_query("飛行車の最高速度は時速150キロメートルです。")
document_2_vector = embeddings.embed_query("鶏肉を適切に下味をつけた後、中火で焼きながらたまに裏返し、外側は香ばしく中は柔らかく仕上げる。")

cos_sim_1 = dot(query_vector, document_1_vector) / (norm(query_vector)*norm(document_1_vector))

print(f"ドキュメント1と質問の類似度：{cos_sim_1}")
cos_sim_2 = dot(query_vector, document_2_vector) / (norm(query_vector)*norm(document_2_vector))

print(f"ドキュメント2と質問の類似度：{cos_sim_2}")