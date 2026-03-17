# rag_ui.py
import gradio as gr
from backend.rag import chain


def ask_question(question, history):
    if not question.strip():
        return "Please enter a question.", history
    answer = chain.invoke(question)
    history.append((question, answer))
    return "", history


iface = gr.ChatInterface(
    fn=ask_question,
    title="IFMIS PDF QA Assistant",
    description="Ask questions about your IFMIS PDF. Powered by Ollama, LangChain, and Chroma embeddings.",
)

if __name__ == "__main__":
    iface.launch()


# rag_ui_gradio_blocks.py
# import gradio as gr
# from rag import chain

# def ask_question(question, history):
#     if not question.strip():
#         return "", history
#     answer = chain.invoke(question)
#     history.append((question, answer))
#     return "", history

# with gr.Blocks(title="IFMIS PDF QA Assistant") as demo:
#     gr.Markdown("## IFMIS PDF QA Assistant")
#     gr.Markdown("Ask questions about your PDF. Powered by Ollama, LangChain, and Chroma embeddings.")

#     chatbot = gr.Chatbot(elem_id="chatbot", label="Chat History").style(height=500)  # height in px
#     with gr.Row():
#         txt = gr.Textbox(
#             placeholder="Type your question here...",
#             label="Your Question",
#             lines=2,
#             max_lines=5
#         )
#         submit = gr.Button("Send", variant="primary")

#     # Connect input to chatbot
#     txt.submit(ask_question, [txt, chatbot], [txt, chatbot])
#     submit.click(ask_question, [txt, chatbot], [txt, chatbot])

# demo.launch()
