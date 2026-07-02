"""Turn text into vector embeddings using OpenAI's embedding API.

Swap this file out to use a different provider (Anthropic, Cohere, local
sentence-transformers, etc.) — keep the same `embed_texts` / `embed_query`
function signatures and the rest of the pipeline doesn't need to change.
"""
from typing import List

from openai import OpenAI

import config

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Returns one vector per input text."""
    if not texts:
        return []
    client = _get_client()
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
