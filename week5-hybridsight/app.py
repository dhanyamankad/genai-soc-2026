# app.py
import base64
import io
import uuid
import gradio as gr
from PIL import Image

from agent import run_agent_with_trace
from tools_rag import index_documents, load_existing_store
from tools_vision import set_current_image, clear_current_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_pil_image(img: Image.Image) -> str:
    """Base64-encodes a PIL image as a JPEG data URI for the vision tool."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def handle_pdf_upload(files):
    if not files:
        return "⚠️ Please upload at least one PDF."
    pdf_paths = [f.name for f in files]
    try:
        chunk_count = index_documents(pdf_paths)
    except Exception as e:
        return f"⚠️ Indexing failed: {e}"
    if chunk_count == 0:
        return "⚠️ Indexed 0 chunks — check the PDF(s) actually contain extractable text."
    return f"✅ {len(pdf_paths)} document(s) indexed — {chunk_count} chunks stored."


def handle_image_upload(img):
    if img is None:
        clear_current_image()
        return "No image uploaded."
    try:
        set_current_image(encode_pil_image(img))
    except Exception as e:
        return f"⚠️ Could not process that image: {e}"
    return "✅ Image ready — ask a question about it below."


def respond(user_input, chat_history, session_id):
    """Called on Submit. Runs the agent and updates the chat + trace panel."""
    if not user_input.strip():
        return chat_history, "", "⚠️ Please enter a question."

    final_answer, trace_log = run_agent_with_trace(user_input, session_id)

    chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": final_answer},
    ]
    trace_text = "\n".join(trace_log) if trace_log else "No tools were called."

    return chat_history, "", trace_text


def create_session_id():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Theme — teal accent (matches the Week 5 / Hybrid Agents brand colour)
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#1a5f70",
    button_primary_background_fill_hover="#154d5c",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e8f6fa",
    button_secondary_text_color="#1a5f70",
    block_title_text_color="#1a1a1a",
    block_label_text_color="#888888",
)

existing_chunks = load_existing_store()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="HybridSight — RAG + Web + Vision Agent", theme=theme) as demo:

    session_id = gr.State(value=create_session_id)

    gr.Markdown("""
    # 🔭 HybridSight — RAG + Web Search + Vision Agent
    A hybrid agent that answers from your uploaded **PDFs** (RAG), the **live web**,
    **Wikipedia**, and uploaded **images** — all in one conversation — with a
    reasoning trace showing exactly which tool handled each part of the answer.
    """)

    with gr.Row():

        # ── Left: knowledge base + image ───────────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("### 🗂 Knowledge Base")
            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload PDF Documents",
            )
            index_btn = gr.Button("⟳  Index Documents", variant="primary")
            doc_status = gr.Textbox(
                label="Document status",
                value=(
                    f"✅ Loaded {existing_chunks} existing chunks from disk."
                    if existing_chunks else "No documents indexed yet."
                ),
                interactive=False,
                lines=1,
            )

            gr.Markdown("### 🖼 Image")
            image_input = gr.Image(type="pil", label="Upload an image")
            image_status = gr.Textbox(
                label="Image status",
                value="No image uploaded.",
                interactive=False,
                lines=1,
            )

            with gr.Accordion("ℹ️ How HybridSight routes questions", open=False):
                gr.Markdown("""
**search_documents** — used when a PDF is indexed and the question sounds document-specific.

**describe_image** — used when you ask about the uploaded image.

**DuckDuckGoSearchRun** — used for current events and anything recent.

**wikipedia_search** — used for general knowledge and historical facts.

The agent picks the tool itself — open the trace panel on the right to see which one(s) it used.
                """)

        # ── Right: chat ─────────────────────────────────────────────────────
        with gr.Column(scale=2):

            gr.Markdown("### 💬 Ask HybridSight")

            chatbot = gr.Chatbot(
                height=440,
                show_label=False,
                placeholder="Upload a PDF or image on the left (optional), then ask anything here.",
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask about your PDF, an image, the web, or general knowledge...",
                    show_label=False,
                    lines=1,
                    scale=5,
                    container=False,
                )
                submit_btn = gr.Button("🔍 Ask", variant="primary", scale=1)

            with gr.Accordion("🔍 Agent Reasoning Trace", open=False):
                trace_output = gr.Textbox(
                    label="Tool calls made by the agent",
                    lines=8,
                    interactive=False,
                    placeholder="Tool call trace will appear here after you ask a question...",
                )

            gr.Examples(
                examples=[
                    ["What does the uploaded PDF say, according to the document?"],
                    ["What's the latest news about ISRO's next launch?"],
                    ["What's in this picture?"],
                    ["Who was Ada Lovelace?"],
                ],
                inputs=user_input,
            )

    gr.Markdown(
        "<center><small>HybridSight · ChromaDB RAG · DuckDuckGo · Wikipedia · "
        "Vision (Groq qwen3.6-27b) · LangGraph · Gradio</small></center>"
    )

    # ── Wiring ────────────────────────────────────────────────────────────────
    index_btn.click(fn=handle_pdf_upload, inputs=[file_input], outputs=[doc_status])

    image_input.upload(fn=handle_image_upload, inputs=[image_input], outputs=[image_status])
    image_input.clear(fn=handle_image_upload, inputs=[image_input], outputs=[image_status])

    submit_btn.click(
        fn=respond,
        inputs=[user_input, chatbot, session_id],
        outputs=[chatbot, user_input, trace_output],
    )
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, session_id],
        outputs=[chatbot, user_input, trace_output],
    )


if __name__ == "__main__":
    demo.launch(share=False)