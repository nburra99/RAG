"""Persist and query embedded chunks using Chroma (local, on-disk vector DB).

To swap in a different vector store (FAISS, Pinecone, Weaviate, ...),
implement a class with the same `add()` and `query()` methods.
"""
from typing import List, Dict, Any

import chromadb

import config
from src.ingestion.chunker import Chunk
from src.embeddings.embedder import embed_texts, embed_query


class VectorStore:
    def __init__(self, persist_dir: str = None, collection_name: str = None):
        self.persist_dir = persist_dir or config.VECTOR_DB_DIR
        self.collection_name = collection_name or config.COLLECTION_NAME

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name
        )

    def add(self, chunks: List[Chunk], batch_size: int = 100) -> None:
        """Embed and store a list of chunks."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = embed_texts([c.text for c in batch])
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source} for c in batch],
            )

    def query(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Return the top_k chunks most similar to query_text."""
        top_k = top_k or config.TOP_K
        query_embedding = embed_query(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            hits.append({"text": doc, "source": meta.get("source"), "distance": dist})

        return hits

    def count(self) -> int:
        return self._collection.count()
