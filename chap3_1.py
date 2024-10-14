from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


pdf_loader = PyPDFLoader("./sample.pdf")
documents = pdf_loader.load()

# text_splitter = SpacyTextSplitter(
#     chunk_size = 300,
#     pipeline="ja_core_news_sm"
# )

# より賢く分割するための分割器
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 分割
split_texts_recursive = recursive_splitter.split_documents(documents)

# 分割結果を確認
for chunk in split_texts_recursive:
    print(chunk)

print(f"分割前のドキュメント数: {len(documents)}")
print(f"分割後のドキュメント数: {len(split_texts_recursive)}")


# =============================================================
from langchain.text_splitter import CharacterTextSplitter

# PDFを読み込む
pdf_loader = PyPDFLoader("./sample.pdf")

# PDFからテキストを抽出
documents = pdf_loader.load()

# 抽出したテキストを確認
for doc in documents:
    print(doc.page_content)

# テキスト分割器の設定（ここでは1000文字ごとに分割）
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 分割されたテキストリストを生成
split_texts = text_splitter.split_documents(documents)

# 分割されたテキストを確認
for chunk in split_texts:
    print(chunk)