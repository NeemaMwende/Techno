# import phoenix as px
# from phoenix.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor

import bs4
# from langchainhub import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama

import os

from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
tracer_provider = register(
    space_id=os.getenv("ARIZE_SPACE_ID"),
    api_key=os.getenv("ARIZ_KEY"),
    project_name="Technobrain",
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# -------------------------
# Phoenix Monitoring Setup
# -------------------------

# px.launch_app()  # Launch Phoenix UI

# tracer_provider = register(
#     project_name="default",
#     endpoint="http://0.0.0.0:6006",
#     auto_instrument=True
# )

# LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

os.environ["USER_AGENT"] = "rag-bot/1.0"

# -------------------------
# LLM
# -------------------------

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0
)

# -------------------------
# Load Data
# -------------------------

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)

docs = loader.load()

# -------------------------
# Split Documents
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)

# -------------------------
# Embeddings
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Vector Store
# -------------------------

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model
)

# Retrieve more docs initially for reranking
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# -------------------------
# Reranker
# -------------------------

reranker = CrossEncoder("BAAI/bge-reranker-base")


def rerank_documents(question, docs, top_k=3):
    """Rerank retrieved documents using a cross-encoder."""

    if not docs:
        return []

    pairs = [[question, doc.page_content] for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]


# -------------------------
# Prompt
# -------------------------

# prompt = hub.pull("rlm/rag-prompt")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use the following pieces of context to answer the question at the end.
If you do not know the answer, say you do not know.

Context:
{context}

Question:
{question}

Answer:
""")

# -------------------------
# Helper Function
# -------------------------

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


def retrieve_and_rerank(question):
    """Retrieve documents then rerank them."""

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant context found."

    reranked_docs = rerank_documents(question, docs, top_k=3)

    return "\n\n".join(doc.page_content for doc in reranked_docs)


# -------------------------
# RAG Chain
# -------------------------

rag_chain = (
    {
        "context": retrieve_and_rerank,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------
# Interactive Chat Loop
# -------------------------

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() in ["exit", "quit"]:
        break

    result = rag_chain.invoke(question)

    print("\nAnswer:")
    print(result)