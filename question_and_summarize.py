import os
from dotenv import load_dotenv
load_dotenv()
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

import tiktoken

tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer

tokenizer.encode("this is a pen")
tokenizer.encode("this")
len(tokenizer.encode("this is a pen."))

import PyPDF2
from langchain.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("https://arxiv.org/pdf/2210.03629.pdf")

arxiv_by_pdf_loader = pdf_loader.load()
arxiv_by_pdf_loader