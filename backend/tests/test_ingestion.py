import pymupdf
import pytest

from app.ingestion import extract_pdf_pages, chunk_pages


def create_pdf(path, page_texts):
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_missing_pdf_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_pdf_pages(tmp_path / "missing.pdf")


def test_textless_pdf_raises_value_error(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    create_pdf(pdf_path, ["", ""])

    with pytest.raises(ValueError, match="No extractable text"):
        extract_pdf_pages(pdf_path)


def test_extract_pdf_preserves_source_page_and_hash(tmp_path):
    pdf_path = tmp_path / "notes.pdf"
    create_pdf(pdf_path, ["First page notes", "", "Third page notes"])

    pages = extract_pdf_pages(pdf_path)

    assert len(pages) == 2
    assert pages[0].page_content == "First page notes"
    assert pages[0].metadata["source"] == "notes.pdf"
    assert pages[0].metadata["page"] == 1
    assert pages[1].metadata["page"] == 3
    assert pages[0].metadata["file_sha256"]
    assert pages[0].metadata["file_sha256"] == pages[1].metadata["file_sha256"]


def test_chunk_pages_returns_empty_for_no_pages():
    assert chunk_pages([]) == []
