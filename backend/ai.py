# import phoenix as px
# from phoenix.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor

import bs4
# from langchainhub import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

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

#s.environ["USER_AGENT"] = "rag-bot/1.0"

# -------------------------
# LLM
# -------------------------

llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)


# -------------------------
# Load Data
# -------------------------

# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     )
# )

# docs = loader.load()

loader = DirectoryLoader(
    "documents",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
# doc.metebase["source"] = documents/report1.pdf

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
# Query Rewriting Prompt
# -------------------------

query_rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search assistant.

Given the conversation history and the latest user question,
rewrite the question into a standalone search query that can
be understood without the conversation.

Conversation History:
{history}

User Question:
{question}

Standalone Search Query:
""")

# -------------------------
# Query Rewriter Chain
# -------------------------

query_rewriter = (
    {
        "question": lambda x: x["question"],
        "history": lambda x: x["history"]
    }
    | query_rewrite_prompt
    | llm
    | StrOutputParser()
)

# -------------------------
# Helper Function
# -------------------------

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


def retrieve_and_rerank(inputs):
    """
    Retrieves documents using vector and BM25 retrievers, reranks them using a cross-encoder,
    and formats the top-k documents with sources for RAG.
    """
    question = inputs["question"]
    history = inputs["history"]

    # Rewrite the question using conversation history
    rewritten_query = query_rewriter.invoke({
        "question": question,
        "history": history
    })

    print("\nRewritten Query:", rewritten_query)

    # Retrieve documents
    vector_docs = retriever.invoke(rewritten_query)
    bm25_docs = bm25_retriever.invoke(rewritten_query)

    # Combine and deduplicate documents by content
    all_docs = vector_docs + bm25_docs
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    # Rerank the top documents
    reranked_docs = rerank_documents(rewritten_query, unique_docs, top_k=3)

    # Format context for the LLM
    context = "\n\n".join(doc.page_content for doc in reranked_docs)

    # Format sources for transparency
    sources = "\n".join(
        f"Source {i+1}: {doc.metadata.get('source','unknown')}"
        for i, doc in enumerate(reranked_docs)
    )

    return context + "\n\nSources:\n" + sources

# RAG Chain
# -------------------------

rag_chain = (
    {
        "context": retrieve_and_rerank,
        "question": lambda x: x["question"],
        "history": lambda x: x["history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)
# -------------------------
# Interactive Chat Loop
# -------------------------
chat_history = []

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() in ["exit", "quit"]:
        break

    result = rag_chain.invoke({
        "question": question,
        "history": chat_history
    })

    print("\nAnswer:")
    print(result)

    chat_history.append({
        "question": question,
        "answer": result
    })