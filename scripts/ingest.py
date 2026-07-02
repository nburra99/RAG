#!/usr/bin/env python
"""CLI: build the vector index from documents in data/raw/.

Usage:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import build_index


if __name__ == "__main__":
    build_index()
