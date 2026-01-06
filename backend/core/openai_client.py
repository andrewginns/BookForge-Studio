"""
OpenAI Client Helpers

Shared helpers for invoking OpenAI Chat Completions from backend workflows.
"""

import os
from typing import List, Dict, Optional

from openai import OpenAI


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")
DEFAULT_OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI requests")
    return OpenAI(api_key=resolved_key)


def create_chat_completion(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_OPENAI_MODEL,
    reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
    temperature: Optional[float] = None,
) -> str:
    client = get_openai_client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
    return response.choices[0].message.content
