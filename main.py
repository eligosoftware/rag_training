from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Load all documents from an archive folder
loader = DirectoryLoader(
    "./archive", 
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader
)
documents = loader.load()

# Convert text into vector embeddings
vectorstore = Chroma.from_documents(documents, embedding=OllamaEmbeddings(model="mistral"))

# Initialize Ollama model
llm = OllamaLLM(model="mistral")

# Create Retrieval-Based QA system
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask a question
response = qa.run("What had Oliver found behind the hidden door?")
print(response)
