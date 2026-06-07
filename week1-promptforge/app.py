from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

personas = {

    "Technical Explainer": {
        "system_prompt": (
            "You are a Technical Explainer. Your job is to explain complex topics "
            "in a clear, simple, and jargon-free way. Use analogies and real-world "
            "examples. Always structure your response with a one-line summary first, "
            "then a detailed explanation. Avoid unnecessary complexity."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "What is an API?"
            },
            {
                "role": "assistant",
                "content": (
                    "**Summary:** An API is a messenger that lets two apps talk to each other.\n\n"
                    "**Explanation:** Imagine you're at a restaurant. You (the app) tell the waiter "
                    "(the API) your order, and the waiter goes to the kitchen (the server) and brings "
                    "back your food (the data). You never go into the kitchen yourself — the waiter "
                    "handles all the communication. That's exactly what an API does between software systems."
                )
            }
        ],
        "output_format": "text"
    },

    "Debate Coach": {
        "system_prompt": (
            "You are a Debate Coach. For any topic the user gives you, you must argue "
            "BOTH sides — first the 'For' side, then the 'Against' side. Be persuasive, "
            "balanced, and logical. Use bullet points for each side. End with a one-line "
            "neutral summary. Never take a personal stance."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Social media is good for society."
            },
            {
                "role": "assistant",
                "content": (
                    "**✅ For:**\n"
                    "- Connects people across the globe instantly\n"
                    "- Gives a voice to marginalized communities\n"
                    "- Accelerates the spread of important information\n\n"
                    "**❌ Against:**\n"
                    "- Promotes misinformation and echo chambers\n"
                    "- Linked to anxiety, depression, and low self-esteem\n"
                    "- Enables surveillance and data exploitation\n\n"
                    "**Neutral Summary:** Social media is a powerful tool — its impact depends entirely on how it is used."
                )
            }
        ],
        "output_format": "text"
    },

    "Code Reviewer": {
        "system_prompt": (
            "You are a strict but helpful Code Reviewer. When given code, analyze it and "
            "respond ONLY with a valid JSON object — no extra text, no markdown fences. "
            "The JSON must have exactly these three keys:\n"
            "- 'issues': a list of strings describing problems found\n"
            "- 'suggestions': a list of strings with improvement ideas\n"
            "- 'severity': one of 'low', 'medium', or 'high'\n"
            "If the code looks fine, return empty lists for issues and suggestions, "
            "and 'low' for severity."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "def add(a, b):\n    return a + b\nprint(add(2, '3'))"
            },
            {
                "role": "assistant",
                "content": (
                    '{"issues": ["Type mismatch: adding int and string will cause a TypeError at runtime"], '
                    '"suggestions": ["Add type hints: def add(a: int, b: int) -> int", '
                    '"Validate input types before performing the operation"], '
                    '"severity": "high"}'
                )
            }
        ],
        "output_format": "json"
    },

    "Creative Writer": {
        "system_prompt": (
            "You are a Creative Writer with a vivid, expressive style. When given any "
            "topic or prompt, respond with rich narrative prose. Use sensory details, "
            "metaphors, and emotion. Your writing should feel alive and cinematic. "
            "Avoid dry or factual language — every response should read like a page "
            "from a novel."
        ),
        "few_shot_examples": [
            {
                "role": "user",
                "content": "Describe rain."
            },
            {
                "role": "assistant",
                "content": (
                    "The rain didn't fall — it descended like a curtain being drawn across the sky, "
                    "each drop a tiny drumbeat on the rooftops, whispering secrets to the concrete below. "
                    "The air tasted of wet earth and forgotten summers, and the street lights bled amber "
                    "halos into the gathering puddles, turning the mundane world into something almost mythical."
                )
            }
        ],
        "output_format": "text"
    }

}


def build_messages(persona_name, user_message):
    """
    Builds the full message list to send to the AI.
    Includes the few-shot examples followed by the real user message.
    """
    persona = personas[persona_name]

    # start with the few-shot examples for this persona
    messages = list(persona["few_shot_examples"])

    # add the actual user message at the end
    messages.append({
        "role": "user",
        "content": user_message
    })

    return messages


def chat_with_persona(persona_name, user_message, temperature=0.7):
    """
    Sends the message to Groq API with the selected persona's
    system prompt and few-shot examples. Streams the response
    back token by token.
    """

    # initialize the Groq client using the API key from .env
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # get the persona config
    persona = personas[persona_name]

    # build the message list (few-shots + real message)
    few_shot_and_user = build_messages(persona_name, user_message)

    # system prompt goes as the FIRST message in the list
    messages = [
        {
            "role": "system",
            "content": persona["system_prompt"]
        }
    ] + few_shot_and_user     

    # make the streaming API call
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=temperature,
        stream=True
    )

    # yield accumulated text token by token
    accumulated = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            accumulated += token
            yield accumulated

# Quick test — print all persona names

import json
import gradio as gr

# ─────────────────────────────────────────
# JSON rendering for Code Reviewer
# ─────────────────────────────────────────

def render_response(persona_name, raw_text):
    """
    If persona is Code Reviewer, parse JSON and render it nicely.
    Otherwise return the text as-is.
    """
    if persona_name != "Code Reviewer":
        return raw_text

    try:
        data = json.loads(raw_text)
        issues = data.get("issues", [])
        suggestions = data.get("suggestions", [])
        severity = data.get("severity", "unknown")

        # build a clean markdown string from the JSON fields
        output = f"### 🔍 Code Review Results\n\n"
        output += f"**Severity:** `{severity.upper()}`\n\n"

        output += "**Issues Found:**\n"
        if issues:
            for issue in issues:
                output += f"- ❌ {issue}\n"
        else:
            output += "- ✅ No issues found\n"

        output += "\n**Suggestions:**\n"
        if suggestions:
            for suggestion in suggestions:
                output += f"- 💡 {suggestion}\n"
        else:
            output += "- ✅ No suggestions\n"

        return output

    except json.JSONDecodeError:
        return f"⚠️ Could not parse JSON response. Raw output:\n\n{raw_text}"


# ─────────────────────────────────────────
# Main chat handler for Gradio
# ─────────────────────────────────────────

def respond(user_message, chat_history, persona_name, temperature):
    """
    Called every time the user submits a message.
    Streams the response into the Gradio chatbot.
    """
    if not user_message.strip():
        yield chat_history
        return

    # add user message to history
    chat_history = chat_history + [{"role": "user", "content": user_message}]
    # add empty assistant message that we'll fill in
    chat_history = chat_history + [{"role": "assistant", "content": ""}]

    # stream the response token by token
    for partial in chat_with_persona(persona_name, user_message, temperature):
        chat_history[-1]["content"] = render_response(persona_name, partial)
        yield chat_history


# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────

def update_system_prompt(persona_name):
    """Returns the system prompt for the selected persona."""
    return personas[persona_name]["system_prompt"]


with gr.Blocks(title="PromptForge") as app:

    gr.Markdown("# 🔧 PromptForge — Multi-Mode AI Assistant")
    gr.Markdown("Pick a persona, type your message, and watch the AI respond in character.")

    with gr.Row():

        # left column — controls
        with gr.Column(scale=1):
            persona_dropdown = gr.Dropdown(
                choices=list(personas.keys()),
                value="Technical Explainer",
                label="🎭 Select Persona"
            )
            temperature_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="🌡️ Temperature (creativity)"
            )
            with gr.Accordion("📋 Active System Prompt", open=False):
                system_prompt_display = gr.Textbox(
                    value=personas["Technical Explainer"]["system_prompt"],
                    label="",
                    lines=6,
                    interactive=False
                )

        # right column — chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=450,
                render_markdown=True,
                placeholder="Your conversation will appear here..."
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send ➤", scale=1, variant="primary")

            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    # ── wire everything together ──

    persona_dropdown.change(
        fn=update_system_prompt,
        inputs=persona_dropdown,
        outputs=system_prompt_display
    )

    send_btn.click(
        fn=respond,
        inputs=[user_input, chatbot, persona_dropdown, temperature_slider],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        outputs=user_input
    )

    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, persona_dropdown, temperature_slider],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        outputs=user_input
    )

    clear_btn.click(fn=lambda: [], outputs=chatbot)


if __name__ == "__main__":
    app.launch()


if __name__ == "__main__":
    app.launch()
