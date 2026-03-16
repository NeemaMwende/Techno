import gradio as gr

# Chat history to preserve conversation
chat_history = []

def chat_fn(user_question):
    """
    Gradio callback for the RAG chatbot.
    """
    if not user_question.strip():
        return "", chat_history

    # Get response from RAG chain
    answer = rag_chain.invoke({
        "question": user_question,
        "history": chat_history
    })

    # Append to history
    chat_history.append({
        "question": user_question,
        "answer": answer
    })

    # Format chat for display (optional: you can format nicely)
    formatted_history = [(h["question"], h["answer"]) for h in chat_history]

    return "", formatted_history

# Build Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("## RAG Chatbot with Re-ranking & History")
    chat = gr.Chatbot()
    user_input = gr.Textbox(
        placeholder="Type your question here...",
        show_label=False
    )
    send_btn = gr.Button("Send")

    send_btn.click(
        fn=chat_fn,
        inputs=user_input,
        outputs=[user_input, chat]
    )

    # Optional: allow pressing Enter to send
    user_input.submit(
        fn=chat_fn,
        inputs=user_input,
        outputs=[user_input, chat]
    )

# Launch the Gradio app
iface.launch()
