from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

pdf_loader = PyPDFLoader("ifmis.pdf")
documents = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ".", " ", ""],
    chunk_size=500,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)

# Local embedding model (MiniLM)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If you do not know the answer, say you do not know.

Context:
{context}

Question:
{question}
""")

llm = ChatOllama(
    model="qwen3:1.8b",
    temperature=0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("what is the education background of Neema")

print(result)