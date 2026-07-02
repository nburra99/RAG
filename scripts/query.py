#!/usr/bin/env python
"""CLI: ask a question against the indexed documents.

Usage:
    python scripts/query.py "What does the document say about X?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import answer_question


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/query.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer = answer_question(question)
    print("\n" + answer + "\n")
