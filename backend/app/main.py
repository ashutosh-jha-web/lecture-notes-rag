
import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    BACKEND_DIR,
    CHAT_MODEL,
    EVAL_LOG_PATH,
    FAISS_DIR,
    MAX_UPLOAD_MB,
    UPLOAD_DIR,
)
from app.evaluation import (
    append_evaluation,
    create_evaluation_record,
    update_evaluation,
    evaluation_lock,
)
from app.ragas_evaluator import aevaluate_rag_response
from app.ingestion import extract_pdf_pages, chunk_pages
from app.query_processing import route_query, decompose_query
from app.query_router import is_greeting, greeting_response
from app.rag_chain import (
    answer_question,
    extract_citations,
    rewrite_question,
    stream_answer,
)
from app.retrieval import make_reranked_retriever
from app.vector_store import build_and_save_index, resolve_index_dir


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lecture Notes Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

document_lock = threading.RLock()
active_document = {
    "document_loaded": False,
    "filename": None,
    "filenames": [],
    "pages_extracted": 0,
    "chunks_created": 0,
    "store": None,
}


class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=10,
    )


def history_as_dicts(history: list[ConversationMessage]) -> list[dict]:
    return [
        {"role": message.role, "content": message.content}
        for message in history
    ]


def resolve_backend_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else BACKEND_DIR / path


def get_latest_evaluation() -> dict:
    path = resolve_backend_path(EVAL_LOG_PATH)

    if not path.exists():
        return {
            "latest": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            },
            "status": "no_evaluations",
        }

    try:
        with evaluation_lock:
            with path.open("r", encoding="utf-8") as file:
                for line in reversed(file.readlines()):
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    metrics = record.get("metrics", {})

                    return {
                        "latest": {
                            "faithfulness": metrics.get("faithfulness"),
                            "answer_relevancy": metrics.get("answer_relevancy"),
                            "context_precision": metrics.get("context_precision"),
                        },
                        "status": record.get("status", "unknown"),
                    }

    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read the evaluation log.")
        raise HTTPException(
            status_code=500,
            detail="Could not read evaluation results.",
        )

    return {
        "latest": {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        },
        "status": "no_evaluations",
    }


async def run_ragas_evaluation(
    record: dict,
    question: str,
    answer: str,
    contexts: list[str],
) -> None:
    """Evaluate a completed RAG answer after the HTTP response."""
    try:
        if not contexts:
            raise ValueError(
                "No non-empty retrieved contexts were available."
            )

        scores, metric_errors = await aevaluate_rag_response(
            question=question,
            answer=answer,
            contexts=contexts,
            include_errors=True,
        )
        record["metrics"] = {
            "faithfulness": scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
            "context_precision": scores.get("context_precision"),
        }
        if metric_errors:
            record["metric_errors"] = metric_errors
        else:
            record["metric_errors"] = {}

        if scores and all(score is not None for score in scores.values()):
            record["status"] = "completed"
            record["error"] = None
        elif scores and any(score is not None for score in scores.values()):
            record["status"] = "partial"
            record["error"] = (
                "Some RAGAS metrics could not be calculated."
            )
        else:
            record["status"] = "failed"
            record["error"] = (
                "; ".join(
                    f"{name}: {message}"
                    for name, message in metric_errors.items()
                )
                if metric_errors
                else "No RAGAS metrics could be calculated."
            )

    except Exception as exc:
        logger.exception("Background RAGAS evaluation failed.")
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"

    try:
        update_evaluation(record)
        logger.info(
             "[EVALUATION] Updated record %s with status=%s",
            record.get("evaluation_id"),
            record.get("status", "unknown"),
        )
    except Exception:
        logger.exception("Could not update the final evaluation record.")


def deduplicate_documents(docs: list) -> list:
    """
    Remove duplicate chunks while preserving retrieval order.

    Source, page, and chunk ID are preferred for identity. The text
    itself is used as a fallback when metadata is incomplete.
    """
    unique_docs = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata or {}

        source = metadata.get("source")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")

        if chunk_id is not None:
            key = ("metadata", source, page, chunk_id)
        else:
            key = (
                "content",
                source,
                page,
                (doc.page_content or "").strip(),
            )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def retrieve_and_rerank(
    store,
    question: str,
    history: list[dict] | None = None,
) -> tuple[list, str, list[str]]:
    """
    Rewrite the question when needed, retrieve and rerank documents,
    and decompose multi-part questions into separate retrieval queries.

    Returns:
        documents: Combined, deduplicated document chunks.
        search_question: Rewritten standalone search question.
        subquestions: Queries actually used for retrieval.
    """
    start = time.perf_counter()
    search_question = rewrite_question(question, history)

    logger.info(
        "[FOLLOW-UP] Original question: %s | Search question: %s",
        question,
        search_question,
    )
    logger.info(
        "[TIMING] Question rewriting: %.2f seconds",
        time.perf_counter() - start,
    )

    # Check both the original and rewritten question so rewriting does not
    # accidentally hide a multi-part request from the decomposition logic.
    original_route = route_query(question)
    search_route = route_query(search_question)

    if original_route == "decompose" or search_route == "decompose":
        subquestions = decompose_query(search_question)

        # Protect against an empty decomposition result.
        if not subquestions:
            subquestions = [search_question]

        # Avoid excessive retrieval calls for unusually long queries.
        max_subquestions = 4
        if len(subquestions) > max_subquestions:
            logger.warning(
                "[DECOMPOSE] Received %d subquestions; limiting to %d.",
                len(subquestions),
                max_subquestions,
            )
            subquestions = subquestions[:max_subquestions]

        route = "decompose"
    else:
        subquestions = [search_question]
        route = original_route

    logger.info(
        "[DECOMPOSE] Route=%s | Retrieval queries=%d",
        route,
        len(subquestions),
    )

    for index, subquestion in enumerate(subquestions, start=1):
        logger.info(
            "[DECOMPOSE] Subquestion %d/%d: %s",
            index,
            len(subquestions),
            subquestion,
        )

    # Create the reranked retriever once and reuse it for each subquestion.
    start = time.perf_counter()
    retriever = make_reranked_retriever(store)

    logger.info(
        "[TIMING] Retriever creation: %.2f seconds",
        time.perf_counter() - start,
    )

    combined_docs = []

    for index, subquestion in enumerate(subquestions, start=1):
        start = time.perf_counter()
        docs = retriever.invoke(subquestion)

        logger.info(
            "[RETRIEVAL] Query %d/%d returned %d reranked chunks "
            "in %.2f seconds.",
            index,
            len(subquestions),
            len(docs),
            time.perf_counter() - start,
        )

        combined_docs.extend(docs)

    # Preserve retrieval order while removing duplicate chunks.
    documents = deduplicate_documents(combined_docs)

    logger.info(
        "[RETRIEVAL] Total chunks before deduplication=%d; "
        "after deduplication=%d.",
        len(combined_docs),
        len(documents),
    )

    return documents, search_question, subquestions


@app.get("/")
def root():
    return {"message": "Lecture Notes Q&A API is running."}


@app.get("/status")
def get_status():
    with document_lock:
        return {
            "document_loaded": active_document["document_loaded"],
            "filename": active_document["filename"],
            "filenames": list(active_document["filenames"]),
            "pages_extracted": active_document["pages_extracted"],
            "chunks_created": active_document["chunks_created"],
        }


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload multiple PDFs in one request.
    This batch replaces the currently active document collection.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Select at least one PDF.",
        )

    upload_dir = resolve_backend_path(UPLOAD_DIR)
    target_index = resolve_index_dir(FAISS_DIR)

    upload_dir.mkdir(parents=True, exist_ok=True)
    target_index.parent.mkdir(parents=True, exist_ok=True)

    temp_pdf_paths: list[Path] = []
    temp_index = None
    backup_index = None
    uploaded_names: list[str] = []
    all_chunks = []
    total_pages = 0

    try:
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="Every uploaded file must have a filename.",
                )

            filename = Path(file.filename).name

            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are accepted: {filename}",
                )

            total_bytes = 0

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".pdf",
                dir=upload_dir,
                delete=False,
            ) as temp_file:
                temp_pdf = Path(temp_file.name)
                temp_pdf_paths.append(temp_pdf)

                while True:
                    data = await file.read(1024 * 1024)
                    if not data:
                        break

                    total_bytes += len(data)

                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"{filename} exceeds the "
                                f"{MAX_UPLOAD_MB} MB per-file upload limit."
                            ),
                        )

                    temp_file.write(data)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"The uploaded file is empty: {filename}",
                )

            with temp_pdf.open("rb") as pdf_file:
                if pdf_file.read(5) != b"%PDF-":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid PDF file: {filename}",
                    )

            pages = extract_pdf_pages(temp_pdf)

            for page in pages:
                page.metadata["source"] = filename

            chunks = chunk_pages(pages)

            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No readable text chunks were found in {filename}."
                    ),
                )

            uploaded_names.append(filename)
            total_pages += len(pages)
            all_chunks.extend(chunks)

        temp_index = Path(
            tempfile.mkdtemp(
                prefix="faiss_staging_",
                dir=target_index.parent,
            )
        )

        new_store = build_and_save_index(
            all_chunks,
            index_dir=temp_index,
        )

        with document_lock:
            if target_index.exists():
                backup_index = target_index.with_name(
                    target_index.name + "_backup"
                )

                if backup_index.exists():
                    shutil.rmtree(backup_index)

                target_index.rename(backup_index)

            try:
                temp_index.rename(target_index)
                temp_index = None
            except Exception:
                if backup_index and backup_index.exists():
                    backup_index.rename(target_index)
                    backup_index = None
                raise

            for temp_pdf, filename in zip(
                temp_pdf_paths,
                uploaded_names,
            ):
                shutil.copy2(temp_pdf, upload_dir / filename)

            active_document.update(
                {
                    "document_loaded": True,
                    "filename": ", ".join(uploaded_names),
                    "filenames": list(uploaded_names),
                    "pages_extracted": total_pages,
                    "chunks_created": len(all_chunks),
                    "store": new_store,
                }
            )

            if backup_index and backup_index.exists():
                shutil.rmtree(backup_index)
                backup_index = None

        return {
            "filenames": uploaded_names,
            "filename": ", ".join(uploaded_names),
            "files_uploaded": len(uploaded_names),
            "pages_extracted": total_pages,
            "chunks_created": len(all_chunks),
            "status": "ready",
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("PDF upload or combined indexing failed.")
        raise HTTPException(
            status_code=500,
            detail="Could not process the PDFs. Check the backend logs.",
        )
    finally:
        for file in files:
            await file.close()

        for temp_pdf in temp_pdf_paths:
            if temp_pdf.exists():
                temp_pdf.unlink()

        if temp_index and temp_index.exists():
            shutil.rmtree(temp_index, ignore_errors=True)


@app.post("/ask")
async def ask_question(
    request: AskRequest,
    background_tasks: BackgroundTasks,
):
    question = request.question.strip()
    history = history_as_dicts(request.history)

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be blank.",
        )

    route = route_query(question)

    if is_greeting(question) or route == "greeting":
        return {
            "answer": greeting_response(question),
            "sources": [],
            "route": "greeting",
            "evaluation": None,
        }

    with document_lock:
        if not active_document["document_loaded"]:
            raise HTTPException(
                status_code=409,
                detail="Upload a PDF before asking a question.",
            )

        store = active_document["store"]

    try:
        start = time.perf_counter()

        docs, search_question, subquestions = retrieve_and_rerank(
            store=store,
            question=question,
            history=history,
        )

        logger.info(
            "[TIMING] Complete retrieval pipeline: %.2f seconds",
            time.perf_counter() - start,
        )

        start = time.perf_counter()
        result = answer_question(question, docs, history)

        logger.info(
            "[TIMING] Answer generation: %.2f seconds",
            time.perf_counter() - start,
        )

    except Exception:
        logger.exception("Question answering failed.")

        error_record = create_evaluation_record(
            question=question,
            docs=[],
            status="error",
            error="Question answering failed. Check backend logs.",
        )
        error_record["model"] = CHAT_MODEL

        try:
            append_evaluation(error_record)
        except Exception:
            logger.exception("Could not append the evaluation record.")

        raise HTTPException(
            status_code=502,
            detail="Could not generate an answer. Check backend logs.",
        )

    record = create_evaluation_record(
        question=question,
        answer=result["answer"],
        docs=docs,
        status="pending",
    )
    record["model"] = CHAT_MODEL
    record["status"] = "pending"

    try:
        append_evaluation(record)
    except Exception:
        logger.exception("Could not append the pending evaluation record.")
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "route": route,
            "evaluation": {
                "status": "logging_failed",
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            },
        }
    contexts = [
        doc.page_content
        for doc in docs
        if doc.page_content and doc.page_content.strip()
    ]

    background_tasks.add_task(
        run_ragas_evaluation,
        record,
        question,
        result["answer"],
        contexts,
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "route": route,
        "evaluation": {
            "status": "pending",
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        },
    }


@app.post("/ask/stream")
async def ask_question_stream(
    request: AskRequest,
    background_tasks: BackgroundTasks,
):
    """Stream answer text using server-sent events."""
    question = request.question.strip()
    history = history_as_dicts(request.history)

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be blank.",
        )

    route = route_query(question)

    if is_greeting(question) or route == "greeting":
        greeting = greeting_response(question)

        async def greeting_events():
            yield (
                "event: meta\ndata: "
                + json.dumps({"route": "greeting"})
                + "\n\n"
            )
            yield (
                "event: token\ndata: "
                + json.dumps({"text": greeting}, ensure_ascii=False)
                + "\n\n"
            )
            yield (
                "event: done\ndata: "
                + json.dumps(
                    {
                        "sources": [],
                        "route": "greeting",
                        "evaluation": None,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        return StreamingResponse(
            greeting_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    with document_lock:
        if not active_document["document_loaded"]:
            raise HTTPException(
                status_code=409,
                detail="Upload a PDF before asking a question.",
            )

        store = active_document["store"]

    try:
        docs, search_question, subquestions = retrieve_and_rerank(
            store=store,
            question=question,
            history=history,
        )
    except Exception as exc:
        logger.exception("Question rewriting or retrieval failed.")
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve lecture context. Check backend logs.",
        ) from exc

    async def answer_events():
        answer_parts = []

        try:
            yield (
                "event: meta\ndata: "
                + json.dumps({"route": route})
                + "\n\n"
            )

            start = time.perf_counter()

            for text_chunk in stream_answer(question, docs, history):
                if not text_chunk:
                    continue

                answer_parts.append(text_chunk)

                yield (
                    "event: token\ndata: "
                    + json.dumps(
                        {"text": text_chunk},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

            answer = "".join(answer_parts)

            logger.info(
                "[TIMING] Streaming answer generation: %.2f seconds",
                time.perf_counter() - start,
            )

            record = create_evaluation_record(
                question=question,
                answer=answer,
                docs=docs,
                status="pending",
            )
            record["model"] = CHAT_MODEL
            record["status"] = "pending"

            try:
                append_evaluation(record)
            except Exception:
                logger.exception(
                    "Could not append the pending evaluation record."
                )
                yield (
                    "event: done\ndata: "
                    + json.dumps(
                        {
                            "sources": extract_citations(docs),
                            "route": route,
                            "evaluation": {
                                "status": "logging_failed",
                                "faithfulness": None,
                                "answer_relevancy": None,
                                "context_precision": None,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                return

            contexts = [
                doc.page_content
                for doc in docs
                if doc.page_content and doc.page_content.strip()
            ]

            background_tasks.add_task(
                run_ragas_evaluation,
                record,
                question,
                answer,
                contexts,
            )

            yield (
                "event: done\ndata: "
                + json.dumps(
                    {
                        "sources": extract_citations(docs),
                        "route": route,
                        "evaluation": {
                            "status": "pending",
                            "faithfulness": None,
                            "answer_relevancy": None,
                            "context_precision": None,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        except Exception:
            logger.exception("Streaming answer generation failed.")

            yield (
                "event: error\ndata: "
                + json.dumps(
                    {
                        "detail": (
                            "Could not finish generating the answer. "
                            "Check backend logs."
                        )
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(
        answer_events(),
        media_type="text/event-stream",
        background=background_tasks,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/evaluation")
def get_evaluation():
    return get_latest_evaluation()
