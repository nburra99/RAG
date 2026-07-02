"""Split documents into overlapping chunks suitable for embedding."""
from dataclasses import dataclass
from typing import List

from src.ingestion.loader import Document


@dataclass
class Chunk:
    """A chunk of text ready to be embedded, tagged with its source."""
    id: str
    text: str
    source: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split `text` into overlapping windows of ~chunk_size characters.

    Splits on paragraph/sentence boundaries where possible so chunks read
    naturally, falling back to a hard character cut for very long runs.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to end on a natural boundary (paragraph, then sentence, then space)
        if end < text_len:
            for boundary in ("\n\n", ". ", "\n", " "):
                idx = text.rfind(boundary, start, end)
                if idx != -1 and idx > start:
                    end = idx + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - chunk_overlap

    return chunks


def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """Chunk a list of Documents, returning flat list of Chunk objects."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(id=f"{doc.source}::{i}", text=piece, source=doc.source)
            )
    return all_chunks
