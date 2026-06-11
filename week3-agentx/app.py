# app.py
import uuid
import gradio as gr
from agent import run_agent_with_trace

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Build the Gradio UI
# ═══════════════════════════════════════════════════════════════════════════

def respond(user_input, chat_history, session_id):
    """Called when user hits Submit. Runs the agent and updates UI."""

    if not user_input.strip():
        return chat_history, "", "⚠️ Please enter a question."

    # Run the agent
    final_answer, trace_log = run_agent_with_trace(user_input, session_id)

    # Gradio 6.0 format — list of dicts with role/content keys
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": final_answer})

    # Format trace log for display
    trace_text = "\n".join(trace_log) if trace_log else "No tools were called."

    return chat_history, "", trace_text


def create_session_id():
    """Generate a unique session ID for each browser tab."""
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Gradio Blocks UI
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="AgentX — Research Agent") as demo:

    # Hidden state to store unique session ID per tab
    session_id = gr.State(value=create_session_id)

    # Header
    gr.Markdown("""
    # 🤖 AgentX — Research Agent with Memory & Visible Reasoning
    Ask me anything! I'll search the web and Wikipedia, remember our conversation,
    and show you exactly which tools I used.
    """)

    # Chatbot display
    chatbot = gr.Chatbot(
    label="Conversation",
    height=450,
)

    # Input row
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="e.g. What is the latest news about India's space program?",
            label="Your Question",
            scale=8,
            lines=1,
        )
        submit_btn = gr.Button("🔍 Ask", variant="primary", scale=1)

    # Reasoning trace accordion (collapsed by default)
    with gr.Accordion("🧠 Agent Reasoning Trace", open=False):
        trace_output = gr.Textbox(
            label="Tool calls made by the agent",
            lines=8,
            interactive=False,
            placeholder="Tool call trace will appear here after you ask a question...",
        )

    # Example questions
    gr.Examples(
        examples=[
            ["What is the latest news about India's space program?"],
            ["Who was Homi J. Bhabha?"],
            ["What did ISRO do recently and who founded it?"],
            ["What is today's date?"],
        ],
        inputs=user_input,
    )

    # ── STEP 7 — Wire event handlers ────────────────────────────────────────

    # Submit button click
    submit_btn.click(
        fn=respond,
        inputs=[user_input, chatbot, session_id],
        outputs=[chatbot, user_input, trace_output],
    )

    # Also trigger on Enter key
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, session_id],
        outputs=[chatbot, user_input, trace_output],
    )

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(share=False)