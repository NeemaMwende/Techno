# import phoenix as px
# from phoenix.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor

import bs4 # beautiful soup
# from langchainhub import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults

from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama

from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.tools import tool
from langchain import hub

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

# os.environ["USER_AGENT"] = "rag-bot/1.0"

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

# ranking documents based on exact term matches
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 10

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
# Tools
# -------------------------

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

@tool
def bm25_lookup(question: str) -> str:
    """Use this for keyword-based document search."""
    docs = bm25_retriever.invoke(question)
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def semantic_lookup(question: str) -> str:
    """Use this for semantic similarity-based document search."""
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant context found."
    reranked_docs = rerank_documents(question, docs, top_k=3)
    return "\n\n".join(doc.page_content for doc in reranked_docs)


web_search = TavilySearchResults(max_results=3)


prompt = hub.pull("hwchase17/react")

tools = [semantic_lookup, bm25_lookup, web_search]

agent = create_react_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# -------------------------
# Interactive Chat Loop
# -------------------------

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() in ["exit", "quit"]:
        break

    result = agent_executor.invoke({"input": question})

    print("\nAnswer:", result["output"])
    