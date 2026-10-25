from langchain_core.documents import Document

from app.rag_chain import extract_citations, format_docs, answer_question


def test_extract_citations_preserves_metadata():
    docs = [
        Document(
            page_content="Machine learning learns patterns from data.",
            metadata={
                "source": "Machine_Learning_Detailed_Notes.pdf",
                "page": 2,
                "chunk_id": "CHK-2",
            },
        )
    ]

    citations = extract_citations(docs)

    assert len(citations) == 1
    assert citations[0]["source"] == "Machine_Learning_Detailed_Notes.pdf"
    assert citations[0]["page"] == 2
    assert citations[0]["chunk_id"] == "CHK-2"
    assert citations[0]["excerpt"] == docs[0].page_content


def test_extract_citations_handles_missing_metadata():
    docs = [
        Document(
            page_content="Example content",
            metadata={},
        )
    ]

    citations = extract_citations(docs)

    assert citations[0]["source"] == "Unknown"
    assert citations[0]["page"] is None
    assert citations[0]["chunk_id"] == "Unknown"


def test_extract_citations_limits_excerpt_to_300_characters():
    docs = [
        Document(
            page_content="A" * 500,
            metadata={"source": "notes.pdf", "page": 1, "chunk_id": "CHK-1"},
        )
    ]

    citations = extract_citations(docs)

    assert len(citations[0]["excerpt"]) == 300


def test_format_docs_includes_source_metadata():
    docs = [
        Document(
            page_content="Neural networks learn representations.",
            metadata={"source": "DL_Notes.pdf", "page": 3, "chunk_id": "CHK-7"},
        )
    ]

    formatted = format_docs(docs)

    assert "[source: DL_Notes.pdf, page: 3, chunk: CHK-7]" in formatted
    assert "Neural networks learn representations." in formatted


def test_answer_question_with_no_documents_skips_llm():
    result = answer_question("What is machine learning?", [])

    assert result["answer"] == (
        "I cannot determine the answer from the provided document."
    )
    assert result["sources"] == []


def test_answer_question_with_context_returns_answer_and_sources(monkeypatch):
    from langchain_core.documents import Document
    import app.rag_chain as rag_chain

    docs = [
        Document(
            page_content="TF-IDF represents text using term frequency and inverse document frequency.",
            metadata={"source": "NLP_Notes.pdf", "page": 2, "chunk_id": "CHK-4"},
        )
    ]

    class FakeChain:
        def invoke(self, inputs):
            assert "TF-IDF" in inputs["context"]
            return "The context describes TF-IDF using term frequency and inverse document frequency."

    monkeypatch.setattr(rag_chain, "create_answer_chain", lambda: FakeChain())

    result = rag_chain.answer_question("What is TF-IDF?", docs)

    assert "term frequency" in result["answer"]
    assert result["sources"][0]["source"] == "NLP_Notes.pdf"
    assert result["sources"][0]["page"] == 2
    assert result["sources"][0]["chunk_id"] == "CHK-4"
