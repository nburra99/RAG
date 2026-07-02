"""Retrieve the most relevant chunks for a given query."""
from typing import List, Dict, Any

from src.vectorstore.store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore = None):
        self.store = store or VectorStore()

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Return a list of {text, source, distance} dicts for the query."""
        return self.store.query(query, top_k=top_k)

    @staticmethod
    def format_context(hits: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a single context block with citations."""
        blocks = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(f"[{i}] (source: {hit['source']})\n{hit['text']}")
        return "\n\n".join(blocks)
