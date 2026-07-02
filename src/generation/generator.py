"""Generate a grounded answer from retrieved context using an LLM.

Swap this file to point at a different provider (Anthropic, local model via
Ollama, etc.) — keep the `generate_answer` signature the same.
"""
from openai import OpenAI

import config

_client = None

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the answer isn't in the context, say you don't "
    "know rather than guessing. Cite sources using the [n] markers given "
    "in the context."
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def generate_answer(question: str, context: str) -> str:
    """Call the chat model with the retrieved context and return its answer."""
    client = _get_client()

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
