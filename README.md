# RAG

A modular Retrieval-Augmented Generation (RAG) pipeline in Python.

Ingest documents → chunk → embed → store in a vector database → retrieve relevant
chunks for a query → generate a grounded answer with an LLM.

## Project Structure

```
RAG/
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── src/
│   ├── ingestion/
│   │   ├── loader.py       # Load raw docs (txt, pdf, md) from data/raw
│   │   └── chunker.py      # Split documents into overlapping chunks
│   ├── embeddings/
│   │   └── embedder.py     # Turn text chunks into vector embeddings
│   ├── vectorstore/
│   │   └── store.py        # Persist/query embeddings (Chroma)
│   ├── retrieval/
│   │   └── retriever.py    # Fetch top-k relevant chunks for a query
│   ├── generation/
│   │   └── generator.py    # Call an LLM with retrieved context
│   └── pipeline.py         # Wires everything together
├── scripts/
│   ├── ingest.py           # CLI: build the vector index from data/raw
│   └── query.py            # CLI: ask a question against the index
├── data/
│   ├── raw/                # Put your source documents here
│   └── processed/          # Cached chunked/processed output
└── tests/
    └── test_chunker.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your API key
```

## Usage

1. Drop `.txt`, `.md`, or `.pdf` files into `data/raw/`.
2. Build the index:
   ```bash
   python scripts/ingest.py
   ```
3. Ask a question:
   ```bash
   python scripts/query.py "What does the document say about X?"
   ```

## How it works

- **Ingestion** (`src/ingestion/`) reads raw files and splits them into overlapping
  text chunks so context isn't lost at chunk boundaries.
- **Embeddings** (`src/embeddings/`) converts each chunk into a vector using an
  embedding model (OpenAI by default; swappable).
- **Vector store** (`src/vectorstore/`) persists those vectors in a local Chroma
  database for fast similarity search.
- **Retrieval** (`src/retrieval/`) embeds the incoming query and fetches the
  top-k most similar chunks.
- **Generation** (`src/generation/`) sends the retrieved chunks + question to an
  LLM, which answers using only that context.

## Configuration

All tunables (chunk size, overlap, top-k, model names) live in `config.py` and
can be overridden via environment variables — see `.env.example`.

## Swapping components

- **Different embedding/LLM provider**: edit `src/embeddings/embedder.py` and
  `src/generation/generator.py` — both are thin wrappers, easy to point at
  Anthropic, Cohere, local models (Ollama), etc.
- **Different vector store**: `src/vectorstore/store.py` wraps Chroma; swap in
  FAISS, Pinecone, Weaviate, etc. by implementing the same `add()` / `query()`
  interface.

## Tests

```bash
pytest tests/
```
