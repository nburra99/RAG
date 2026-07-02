"""Wires ingestion, embedding, storage, retrieval, and generation together."""
from pathlib import Path

import config
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.vectorstore.store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import generate_answer


def build_index(raw_dir: Path = None) -> int:
    """Load, chunk, embed, and store all documents in `raw_dir`.

    Returns the number of chunks indexed.
    """
    raw_dir = raw_dir or config.RAW_DATA_DIR

    documents = load_documents(raw_dir)
    if not documents:
        print(f"No supported documents found in {raw_dir}")
        return 0

    chunks = chunk_documents(
        documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    store = VectorStore()
    store.add(chunks)

    print(f"Indexed {len(documents)} document(s) into {len(chunks)} chunk(s).")
    return len(chunks)


def answer_question(question: str, top_k: int = None) -> str:
    """Run the full retrieve -> generate flow for a single question."""
    retriever = Retriever()
    hits = retriever.retrieve(question, top_k=top_k)

    if not hits:
        return "No indexed documents found. Run `python scripts/ingest.py` first."

    context = retriever.format_context(hits)
    return generate_answer(question, context)
