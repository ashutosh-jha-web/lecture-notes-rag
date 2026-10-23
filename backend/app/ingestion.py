
from pathlib import Path
import hashlib

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """Extract text from each PDF page and preserve source metadata."""
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    page_docs: list[Document] = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            page_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                        "file_sha256": file_hash,
                    },
                )
            )

    if not page_docs:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "This PDF may be scanned and require OCR."
        )

    return page_docs


def chunk_pages(page_docs: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks while preserving metadata."""
    if not page_docs:
        return []

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(page_docs)

    for i, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown.pdf"))
        file_hash = str(chunk.metadata.get("file_sha256", ""))

        # Generate a stable, filename-independent source identifier.
        source_id = file_hash[:10] or hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()[:10]

        chunk.metadata["source_id"] = source_id

        # Sequential chunk ID expected by the chunking test.
        chunk.metadata["chunk_id"] = f"CHK-{i + 1}"

    return chunks
