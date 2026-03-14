import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

import bs4
#from langchainhub import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
import os
# -------------------------
# Phoenix Monitoring Setup
# -------------------------

px.launch_app()  # Launch Phoenix UI

tracer_provider = register(
  project_name="default",
  endpoint="http://0.0.0.0:6006",
  auto_instrument=True
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

from phoenix.otel import register


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

retriever = vectorstore.as_retriever(search_kwargs={"k":3})


# -------------------------
# Prompt
# -------------------------

#prompt = hub.pull("rlm/rag-prompt")
prompt = ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end.
If you do not know the answer, say you do not know.

Context:
{context}

Question:
{question}
""")

# -------------------------
# Helper Function
# -------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------
# RAG Chain
# -------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
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
