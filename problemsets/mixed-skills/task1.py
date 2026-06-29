# task1.py
"""
Task 1: Basic LLM Chat

Reads a system prompt and a user prompt (either as command-line arguments or
interactively), calls Groq, and prints the assistant's response.

Note: the brief specifies llama-3.3-70b-versatile, which Groq deprecated on
2026-06-17. This uses openai/gpt-oss-120b, Groq's recommended replacement.
"""
import sys
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"


def get_prompts():
    """Reads system + user prompts from CLI args if given, else interactively."""
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    system_prompt = input("System: ").strip()
    user_prompt = input("User: ").strip()
    return system_prompt, user_prompt


def main():
    system_prompt, user_prompt = get_prompts()

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()