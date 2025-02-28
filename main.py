from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Load all documents from an archive folder
loader = DirectoryLoader("./archive/", glob="**/*.txt")  # Can support PDFs, Word, etc.
documents = loader.load()

# Convert text into vector embeddings
vectorstore = Chroma.from_documents(documents, embedding=OllamaEmbeddings())

# Initialize Ollama model
llm = Ollama(model="mistral")

# Create Retrieval-Based QA system
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask a question
response = qa.run("What are the key points from last month's reports?")
print(response)
