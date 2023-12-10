from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from os import getenv, path
from git import Repo

# Retrieving an API key from environmental variable
api_key = getenv('API')
if api_key is None:
    raise ValueError("Environmental variable `API` doesn't contain the API key!")

# Cloning the repository
repo_path = "./repo"
if not path.exists(repo_path):
    repo = Repo.clone_from("https://github.com/staszczm/AIChallange-A.I.diots", to_path=repo_path)

# Loading the repository
loader = GenericLoader.from_filesystem(
    path=repo_path,
    glob="**/*",  # Recursively find all files within given repo
    suffixes=[".py"],  # Search for Python code files
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=250),
)
documents = loader.load()

# Splitting the files into smaller chunks
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

# Adding split files into embedding database and defining what it returns
db = Chroma.from_documents(texts, OpenAIEmbeddings(openai_api_key=api_key))
retriever = db.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance
    search_kwargs={"k": 10},  # Return the top 10 matches
)

# Initialize LangChain with the OpenAI model
lc = ChatOpenAI(
    model_name='gpt-4-1106-preview',  # Alternative options ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106']
    openai_api_key=api_key,
)

# Memorising the context of previous messages
memory = ConversationSummaryMemory(
    llm=lc, memory_key="chat_history", return_messages=True
)
qa = ConversationalRetrievalChain.from_llm(lc, retriever=retriever, memory=memory)

# Example prompt
question = "What is this project about?"
result = qa(question)
print(f"{question}\n{result['answer']}\n")

# Example prompt
question = "How can I initialize a React agent?"
result = qa(question)
print(f"{question}\n{result['answer']}\n")

# Example prompt
question = "Describe in details what file models.py contains"
result = qa(question)
print(f"{question}\n{result['answer']}\n")
