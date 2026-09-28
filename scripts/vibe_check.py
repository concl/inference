
"""
Test script for the API endpoint.

Contains helper functions for sending prompts and receiving responses, as well as an interactive chat mode.
"""

import os

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "placeholder"
DEFAULT_MODEL = "Qwen3.8-27B"
DEFAULT_SYSTEM = "You are a helpful assistant."


def make_client(base_url=None, api_key=None):
    """Build a client, falling back to OPENAI_BASE_URL / OPENAI_API_KEY."""
    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key or os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY),
    )


def build_messages(prompt, system=DEFAULT_SYSTEM):
    """Wrap a single user prompt in the standard two-message layout."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def ask(client, model, prompt, system=DEFAULT_SYSTEM):
    """One non-streaming request. Returns the reply text."""
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(prompt, system),
    )
    return response.choices[0].message.content


def stream_reply(client, model, prompt, system=DEFAULT_SYSTEM):
    """Yield reply text incrementally as the model produces it."""
    streamer = client.chat.completions.create(
        model=model,
        messages=build_messages(prompt, system),
        stream=True,
    )
    for chunk in streamer:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def interactive(client, model, system=DEFAULT_SYSTEM):
    """Prompt/response loop until the user types 'exit'."""
    print("Entering interactive mode. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user_input.strip().lower() in {"exit", "quit"}:
            return

        print("Assistant: ", end="", flush=True)
        for text in stream_reply(client, model, user_input, system):
            print(text, end="", flush=True)
        print()
