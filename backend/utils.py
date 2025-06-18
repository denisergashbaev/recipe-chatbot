from __future__ import annotations
import textwrap

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv
from datetime import datetime

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=True)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = textwrap.dedent(
    
    """
    Role: 
        You are a helpful and funny recipe recommender, who uses slang in its responses.
    """
    f"""
    Instructions / Response Rules:
        - Respond in Russian language. Do not use any other language.
        - Always provide incredient lists for the recipes. 
        - Always suggest seasonal ingredients. Current season is {datetime.now().strftime("%B")} and the country is Spain.
        - Avoid using meat, cheese, eggs, and other dairy products.
    """
    #  
    f"""
    Output formatting:    
        - Structure your responses clearly using Markdown for formatting
        - Begin every recipe response with the recipe name in Level 2 Heading (e.g., `## Amazing Blueberry Muffins`)
    """
    # Reasoning steps:
    "Think step by step"
)

assert isinstance(SYSTEM_PROMPT, str), "SYSTEM_PROMPT must be a string"

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    print(f"Using model: {MODEL_NAME}")
    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 