# IFMIS PDF QA Assistant

A RAG (Retrieval-Augmented Generation) application that answers questions about IFMIS PDF documents using local LLMs.

## Features

- PDF document loading and chunking
- Vector embeddings with HuggingFace (all-MiniLM-L6-v2)
- Chroma vector store for semantic search
- Ollama LLM (qwen3:1.7b) for question answering
- Gradio web interface

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```
OPENAI_API_KEY=your_key_here
ARIZE_SPACE_ID=your_space_id
ARIZ_KEY=your_key
```

4. Place your PDF as `ifmis.pdf` in the project root.

5. Run the UI:
```bash
python3 rag_ui.py
```

## Requirements

- Python 3.12+
- Ollama installed with qwen3:1.7b model
