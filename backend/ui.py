# rag_ui_streamlit.py
import streamlit as st
from backend.rag import chain

st.set_page_config(page_title="IFMIS PDF QA Assistant", layout="wide")

st.title("IFMIS PDF QA Assistant")
st.markdown("Ask questions about your PDF. Powered by Ollama, LangChain, and Chroma embeddings.")

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input section
question = st.text_area("Your Question", height=80, placeholder="Type your question here...")

if st.button("Send"):
    if question.strip():
        answer = chain.invoke(question)
        st.session_state.history.append((question, answer))
        st.experimental_rerun()

# Display chat history
for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
    st.markdown("---")  # horizontal line
