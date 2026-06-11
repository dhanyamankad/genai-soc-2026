# agent.py
import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════

duckduckgo_tool = DuckDuckGoSearchRun(
    name="DuckDuckGoSearchRun",
    description=(
        "Use for real-time or recent information: current events, latest news, "
        "prices, recent developments, or anything that changes frequently. "
        "Prefer this over Wikipedia for anything after 2022."
    ),
)

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

wikipedia_tool = DuckDuckGoSearchRun(
    name="WikipediaQueryRun",
    description=(
        "Use for background, historical, or encyclopaedic information: "
        "biographies, scientific concepts, historical events, organisations, "
        "definitions, and foundational facts unlikely to change."
    ),
)

@tool
def get_current_date(query: str = "") -> str:
    """Returns today's date. Use when the user asks about 'today',
    'current date', or when you need to know the date to answer accurately."""
    return f"Today's date is {datetime.now().strftime('%A, %B %d, %Y')}."

tools = [duckduckgo_tool, wikipedia_tool, get_current_date]

# ═══════════════════════════════════════════════════════════════════════════
# LLM + MEMORY
# ═══════════════════════════════════════════════════════════════════════════

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0,
    model_kwargs={"parallel_tool_calls": False},
)

memory = MemorySaver()

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

today = datetime.now().strftime("%A, %B %d, %Y")

SYSTEM_PROMPT = f"""You are AgentX, a research assistant. Today is {today}.

Use your tools to answer questions. Always cite which tool gave you the info.
Format answers as: Introduction, Key Facts (with sources), Recent Developments, Conclusion."""

# ═══════════════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════════════

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Streaming trace function
# ═══════════════════════════════════════════════════════════════════════════
def run_agent_with_trace(user_input: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    messages = {"messages": [("user", user_input)]}
    trace_log = []
    final_answer = ""
    seen = set()

    try:
        for event in agent.stream(messages, config=config, stream_mode="updates"):
            for node_name, node_data in event.items():
                for msg in node_data.get("messages", []):
                    msg_type = type(msg).__name__

                    if msg_type == "AIMessage":
                        # Tool calls are in additional_kwargs
                        tool_calls = msg.additional_kwargs.get("tool_calls", [])
                        for tc in tool_calls:
                            name = tc["function"]["name"]
                            args = tc["function"]["arguments"]
                            entry = f"🔧 Tool Called: {name}\n   Input: {args}"
                            if entry not in seen:
                                seen.add(entry)
                                trace_log.append(entry)

                        # Final answer — has content, no tool calls
                        if msg.content and not tool_calls:
                            final_answer = msg.content

                    elif msg_type == "ToolMessage":
                        entry = f"✅ Result from {msg.name}:\n   {str(msg.content)[:200]}"
                        if entry not in seen:
                            seen.add(entry)
                            trace_log.append(entry)

    except Exception as e:
        final_answer = f"⚠️ Error: {str(e)}"
        trace_log.append(f"❌ Exception: {str(e)}")

    return final_answer, trace_log