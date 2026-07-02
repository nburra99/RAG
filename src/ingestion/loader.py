"""Load raw documents (.txt, .md, .pdf) from a directory."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    """A single loaded document before chunking."""
    source: str   # filename, used for citations
    text: str


def _read_txt_or_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(raw_dir: Path) -> List[Document]:
    """Load every supported file in `raw_dir` into a list of Documents."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    documents: List[Document] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = _read_txt_or_md(path)

        text = text.strip()
        if text:
            documents.append(Document(source=path.name, text=text))

    return documents
