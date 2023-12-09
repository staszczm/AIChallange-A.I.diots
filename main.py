from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from os import getenv

api_key = 'key'

# Clone
repo_path = "./repo"
repo = Repo.clone_from("https://github.com/staszczm/AIChallange-A.I.diots", to_path=repo_path)

# Load
loader = GenericLoader.from_filesystem(
    path=repo_path,
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)
documents = loader.load()
print(len(documents))

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
print(len(texts))

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=(), openai_api_key=api_key))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

# Initialize LangChain with the OpenAI model
lc = ChatOpenAI(
    model_name='gpt-4-1106-preview',
    openai_api_key = api_key,
)
memory = ConversationSummaryMemory(
    llm=lc, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(lc, retriever=retriever, memory=memory)

question = "What is this project about?"
result = qa(question)
result["answer"]

question = ("Do you have any idea on how to improve this project?")
result = qa(question)
print(f"""{result["answer"]}""")