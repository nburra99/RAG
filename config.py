"""Central configuration for the RAG pipeline.

All values can be overridden via environment variables (see .env.example).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- Data locations ---
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# --- Models ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120)) # overlap between chunks

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 4))

# --- Vector store ---
VECTOR_DB_DIR = str(BASE_DIR / os.getenv("VECTOR_DB_DIR", "data/processed/chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
