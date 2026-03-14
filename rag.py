from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
# import phoenix as px
# from phoenix.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# tracer_provider = register(
#   project_name="default",
#   endpoint="http://0.0.0.0:6006",
#   auto_instrument=True
# )
# LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

os.environ["OPENAI_API_KEY"] = "OPENAI_KEY_REDACTED "
tracer_provider = register(
    space_id = "ARIZE_SPACE_ID_REDACTED==",
    api_key = "ARIZ_KEY_REDACTED",
    project_name = "Technobrain", # name this to whatever you would like
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

load_dotenv()

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