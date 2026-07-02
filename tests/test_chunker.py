import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_text, chunk_documents
from src.ingestion.loader import Document


def test_short_text_returns_single_chunk():
    text = "This is a short piece of text."
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=120)
    assert chunks == [text]


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=800, chunk_overlap=120) == []


def test_long_text_splits_into_multiple_chunks():
    text = "Sentence number %d. " * 1  # placeholder, built below
    text = " ".join(f"This is sentence {i}." for i in range(200))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200 + 40  # allow boundary slack


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_text("some text", chunk_size=100, chunk_overlap=100)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_chunk_documents_tags_source_and_ids():
    docs = [Document(source="a.txt", text="Hello world, this is document A.")]
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].source == "a.txt"
    assert chunks[0].id == "a.txt::0"
