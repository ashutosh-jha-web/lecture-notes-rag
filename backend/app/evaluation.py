
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime, timezone

from app.config import BACKEND_DIR, EVAL_LOG_PATH


# Shared lock for all evaluation-log reads and writes.
evaluation_lock = threading.RLock()


def resolve_eval_path(path: str | Path = EVAL_LOG_PATH) -> Path:
    """Resolve relative evaluation paths from the backend directory."""
    target = Path(path)

    if not target.is_absolute():
        target = BACKEND_DIR / target

    return target


def append_evaluation(
    record: dict,
    path: str | Path = EVAL_LOG_PATH,
) -> dict:
    """Append one evaluation record as a JSON Lines entry."""
    target = resolve_eval_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    row = dict(record)
    if not row.get("evaluation_id"):
        row["evaluation_id"] = str(uuid.uuid4())
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    with evaluation_lock:
        if target.exists():
            with target.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    existing = json.loads(line)
                    if existing.get("evaluation_id") == row["evaluation_id"]:
                        raise ValueError(
                            "Evaluation ID already exists: "
                            + row["evaluation_id"]
                        )

        with target.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    record["evaluation_id"] = row["evaluation_id"]
    record["timestamp"] = row["timestamp"]

    return row


def update_evaluation(
    record: dict,
    path: str | Path = EVAL_LOG_PATH,
) -> dict:
    """Update an existing JSONL record by evaluation_id."""
    target = resolve_eval_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    evaluation_id = record.get("evaluation_id")
    if not evaluation_id:
        raise ValueError("Cannot update an evaluation without evaluation_id.")

    with evaluation_lock:
        rows = []

        if target.exists():
            with target.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))

        matches = [
            index
            for index, row in enumerate(rows)
            if row.get("evaluation_id") == evaluation_id
        ]

        if not matches:
            raise ValueError(
                "Evaluation record not found: " + evaluation_id
            )

        if len(matches) > 1:
            raise ValueError(
                "Cannot update evaluation with duplicate ID: "
                + evaluation_id
            )

        index = matches[0]
        rows[index] = {
            **rows[index],
            **record,
            "evaluation_id": evaluation_id,
        }

        with target.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        return rows[index]


def create_evaluation_record(
    question: str,
    answer: str = "",
    docs: list | None = None,
    status: str = "pending",
    error: str | None = None,
) -> dict:
    """Create an evaluation record without inventing metric values."""
    docs = docs or []

    sources = []
    contexts = []

    for doc in docs:
        metadata = doc.metadata
        contexts.append(doc.page_content)

        sources.append(
            {
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
            }
        )

    return {
        "evaluation_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
        "metrics": {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        },
        "status": status,
        "error": error,
        "model": None,
        "ragas_version": None,
    }
