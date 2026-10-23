
from langchain_core.documents import Document

from app.ingestion import chunk_pages


def test_chunks_keep_page_and_ids():
    docs = [
        Document(
            page_content="lecture text " * 300,
            metadata={"source": "test.pdf", "page": 3},
        )
    ]

    chunks = chunk_pages(docs)

    assert chunks
    assert all(chunk.metadata["page"] == 3 for chunk in chunks)
    assert all(
        chunk.metadata["chunk_id"].startswith("CHK-")
        for chunk in chunks
    )
    assert all("start_index" in chunk.metadata for chunk in chunks)
