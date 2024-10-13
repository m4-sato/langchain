# from langchain import PromptTemplate
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "{product}はどこの会社が開発した製品ですか？",
#         ),
#         MessagesPlaceholder(variable_name="product"),
#     ]
# )

# print(prompt.format(product="iPhone"))
# print(prompt.format(product="Xpedia"))

# from langchain.prompts import ChatPromptTemplate

# # シンプルなチャットプロンプトテンプレートの作成
# prompt = ChatPromptTemplate.from_template(
#     templete="{product}はどこの会社が開発した製品ですか？",
#     input_variables=[
#         "product"
#     ]
# )

# # テンプレートに製品名を挿入してフォーマット
# print(prompt.format(product="iPhone"))
# print(prompt.format(product="Xpedia"))


# from langchain.prompts import ChatPromptTemplate

# # テンプレートの作成
# prompt = ChatPromptTemplate.from_template(
#     template="{product}はどこの会社が開発した製品ですか？ その製品の主な特徴は何ですか？",
#     input_variables=["product"]
# )

# # フォーマットと出力
# print(prompt.format(product="iPhone"))
# # 出力: iPhoneはどこの会社が開発した製品ですか？ その製品の主な特徴は何ですか？

# print(prompt.format(product="Xpedia"))
# # 出力: Xpediaはどこの会社が開発した製品ですか？ その製品の主な特徴は何ですか？

from langchain.prompts import ChatPromptTemplate

# テンプレートの作成
prompt = ChatPromptTemplate.from_template(
    template="{product}はどこの会社が開発した製品ですか？"
)

# フォーマットと出力
print(prompt.format(product="iPhone"))  # 出力: iPhoneはどこの会社が開発した製品ですか？
print(prompt.format(product="Xpedia"))  # 出力: Xpediaはどこの会社が開発した製品ですか？
