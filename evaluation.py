import os
import openai
import dotenv

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

file = 'OutdoorClothingCatalog_1000.csv'
# loader = CSVLoader(file_path=file)
loader = CSVLoader(file_path=file, encoding='utf-8')
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature = 0.0)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

data[10]

data[11]

### Hard-coded examples

examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

### LLM-Generated examples

from langchain.evaluation.qa import QAGenerateChain


example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())

new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)

new_examples[0]

data[0]

### Combine examples

examples += new_examples

qa.run(examples[0]["query"])

## Manual Evaluation

import langchain
langchain.debug = True

qa.run(examples[0]["query"])

# Turn off the debug mode
langchain.debug = False

## LLM assisted evaluation

predictions = qa.apply(examples)

from langchain.evaluation.qa import QAEvalChain

llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()