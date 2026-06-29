# task3.py
"""
Task 3: One-Tool Agent

A minimal LangGraph ReAct agent with a single tool that adds two integers.
"""
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv()

MODEL = "openai/gpt-oss-120b"


@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two integers and returns the sum."""
    return a + b


def main():
    llm = ChatGroq(model=MODEL, api_key=os.environ.get("GROQ_API_KEY"), temperature=0)
    agent = create_react_agent(model=llm, tools=[add_numbers])

    user_message = "What is 12 + 15?"
    result = agent.invoke({"messages": [("user", user_message)]})

    print(f"User: {user_message}")
    print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()