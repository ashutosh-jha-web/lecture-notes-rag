import json

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client():
    original_state = main.active_document.copy()
    main.active_document.update({
        "document_loaded": False,
        "filename": None,
        "filenames": [],
        "pages_extracted": 0,
        "chunks_created": 0,
        "store": None,
    })

    with TestClient(main.app) as test_client:
        yield test_client

    main.active_document.clear()
    main.active_document.update(original_state)


def test_root_returns_api_message(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Lecture Notes Q&A API is running."


def test_status_reports_no_document(client):
    response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == {
        "document_loaded": False,
        "filename": None,
        "filenames": [],
        "pages_extracted": 0,
        "chunks_created": 0,
    }


def test_ask_rejects_blank_question(client):
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["detail"] == "Question must not be blank."


def test_ask_rejects_question_when_no_document_is_loaded(
    client, monkeypatch
):
    monkeypatch.setattr(main, "route_query", lambda question: "direct")
    monkeypatch.setattr(main, "is_greeting", lambda question: False)

    response = client.post(
        "/ask",
        json={"question": "Explain backpropagation"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Upload a PDF before asking a question."


def test_greeting_returns_without_document_or_evaluation(
    client, monkeypatch
):
    monkeypatch.setattr(main, "route_query", lambda question: "greeting")
    monkeypatch.setattr(main, "is_greeting", lambda question: True)
    monkeypatch.setattr(
        main, "greeting_response", lambda question: "Hello!"
    )

    response = client.post("/ask", json={"question": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Hello!",
        "sources": [],
        "route": "greeting",
        "evaluation": None,
    }


def test_streaming_greeting_returns_sse_events(client, monkeypatch):
    monkeypatch.setattr(main, "route_query", lambda question: "greeting")
    monkeypatch.setattr(main, "is_greeting", lambda question: True)
    monkeypatch.setattr(
        main, "greeting_response", lambda question: "Hello!"
    )

    response = client.post(
        "/ask/stream",
        json={"question": "Hello"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = {}
    current_event = None

    for line in response.text.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
        elif line.startswith("data: ") and current_event:
            events[current_event] = json.loads(line.removeprefix("data: "))

    assert events["meta"] == {"route": "greeting"}
    assert events["token"] == {"text": "Hello!"}
    assert events["done"]["sources"] == []
    assert events["done"]["route"] == "greeting"
    assert events["done"]["evaluation"] is None


def test_upload_rejects_non_pdf_file(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "FAISS_DIR", tmp_path / "faiss_index")

    response = client.post(
        "/upload",
        files={"files": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are accepted: notes.txt"


def test_upload_rejects_empty_pdf(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "FAISS_DIR", tmp_path / "faiss_index")

    response = client.post(
        "/upload",
        files={"files": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty: empty.pdf"


def test_upload_rejects_invalid_pdf_signature(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "FAISS_DIR", tmp_path / "faiss_index")

    response = client.post(
        "/upload",
        files={"files": ("invalid.pdf", b"not a PDF file", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF file: invalid.pdf"


def test_upload_rejects_file_over_size_limit(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "FAISS_DIR", tmp_path / "faiss_index")
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 4)

    response = client.post(
        "/upload",
        files={"files": ("large.pdf", b"%PDF-12345", "application/pdf")},
    )

    assert response.status_code == 413
    assert "exceeds the" in response.json()["detail"]
