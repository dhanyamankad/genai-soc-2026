# task5.py
"""
Task 5: Chatbot with Memory and a Date Tool

A Gradio chat interface backed by a LangGraph ReAct agent with one tool
(get_current_date) and conversation memory via MemorySaver, so follow-up
questions in the same session have context.
"""
import os
import uuid
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

MODEL = "openai/gpt-oss-120b"


@tool
def get_current_date() -> str:
    """Returns today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


llm = ChatGroq(model=MODEL, api_key=os.environ.get("GROQ_API_KEY"), temperature=0)
memory = MemorySaver()

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the get_current_date tool whenever the "
    "user asks about today's date or 'what day is it'. For relative dates "
    "like yesterday or tomorrow, call the tool to get today's date, then "
    "compute the relative date yourself."
)

agent = create_react_agent(model=llm, tools=[get_current_date], checkpointer=memory, prompt=SYSTEM_PROMPT)


def respond(user_input, chat_history, session_id):
    if not user_input.strip():
        return chat_history, ""

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"messages": [("user", user_input)]}, config=config)
    answer = result["messages"][-1].content

    chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, ""


def create_session_id():
    return str(uuid.uuid4())


with gr.Blocks(title="Task 5 — Chatbot with Memory + Date Tool") as demo:
    session_id = gr.State(value=create_session_id)

    gr.Markdown("# 🗓️ Chatbot with Memory and a Date Tool")
    chatbot = gr.Chatbot(height=400, show_label=False)

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask me anything, including what today's date is...",
            show_label=False,
            scale=5,
            container=False,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    submit_btn.click(fn=respond, inputs=[user_input, chatbot, session_id], outputs=[chatbot, user_input])
    user_input.submit(fn=respond, inputs=[user_input, chatbot, session_id], outputs=[chatbot, user_input])


if __name__ == "__main__":
    demo.launch(share=False)