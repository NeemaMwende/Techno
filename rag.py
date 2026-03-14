import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
tracer_provider = register(
    space_id=os.getenv("ARIZE_SPACE_ID"),
    api_key=os.getenv("ARIZ_KEY"),
    project_name="Technobrain",
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

pdf_loader = PyPDFLoader("ifmis.pdf")
documents = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ".", " ", ""],
    chunk_size=500,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("./chroma_db"):
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
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
    model="qwen3:1.7b",
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

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() in ["exit", "quit"]:
        break

    result = chain.invoke(question)

    print("\nAnswer:")
    print(result)