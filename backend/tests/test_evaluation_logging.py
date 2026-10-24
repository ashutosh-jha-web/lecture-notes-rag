import json

import pytest

from app.evaluation import (
    append_evaluation,
    create_evaluation_record,
    update_evaluation,
)


def test_append_generates_id_for_empty_id(tmp_path):
    path = tmp_path / "results.jsonl"
    record = {"evaluation_id": "", "question": "test"}

    saved = append_evaluation(record, path=path)

    assert saved["evaluation_id"]
    assert record["evaluation_id"] == saved["evaluation_id"]


def test_append_rejects_duplicate_id(tmp_path):
    path = tmp_path / "results.jsonl"
    record = create_evaluation_record("test question")
    append_evaluation(record, path=path)

    with pytest.raises(ValueError, match="already exists"):
        append_evaluation(record, path=path)

    assert len(path.read_text().splitlines()) == 1


def test_update_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "results.jsonl"
    record = create_evaluation_record("test question")
    append_evaluation(record, path=path)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    record["status"] = "completed"
    with pytest.raises(ValueError, match="duplicate ID"):
        update_evaluation(record, path=path)


def test_update_preserves_id_and_updates_status(tmp_path):
    path = tmp_path / "results.jsonl"
    record = create_evaluation_record("test question")
    append_evaluation(record, path=path)

    record["status"] = "completed"
    updated = update_evaluation(record, path=path)

    assert updated["evaluation_id"] == record["evaluation_id"]
    assert updated["status"] == "completed"
    assert len(path.read_text().splitlines()) == 1
