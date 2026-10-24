
import json
from pathlib import Path

from langchain_core.documents import Document

from app.evaluation import (
    append_evaluation,
    create_evaluation_record,
    update_evaluation,
)


def main():
    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Evaluation logging
    test_log = eval_dir / "test_results.jsonl"

    docs = [
        Document(
            page_content="Machine learning learns patterns from examples.",
            metadata={
                "source": "sample.pdf",
                "page": 1,
                "chunk_id": "CHK-00001",
            },
        )
    ]

    record = create_evaluation_record(
        question="What is machine learning?",
        answer="Machine learning learns patterns from examples.",
        docs=docs,
        status="pending",
    )

    append_evaluation(record, path=test_log)

    with test_log.open("r", encoding="utf-8") as file:
        loaded = json.loads(file.readlines()[-1])

    assert loaded["question"] == "What is machine learning?"
    assert loaded["sources"][0]["page"] == 1
    assert loaded["metrics"]["faithfulness"] is None
    assert loaded["status"] == "pending"
    assert loaded["timestamp"]
    assert loaded["evaluation_id"]

    print("Evaluation logging test passed!")

    # Test 2: Update an evaluation using a separate test log
    update_log = eval_dir / "test_update_results.jsonl"

    update_record = create_evaluation_record(
        question="Test update question?",
        answer="Initial answer",
        docs=docs,
        status="pending",
    )

    append_evaluation(update_record, path=update_log)

    update_record["status"] = "completed"
    update_record["metrics"] = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
    }
    update_record["error"] = None

    updated = update_evaluation(update_record, path=update_log)

    # Verify the updated record was written to disk
    with update_log.open("r", encoding="utf-8") as file:
        saved_updated = json.loads(file.readlines()[-1])

    assert updated["status"] == "completed"
    assert saved_updated["status"] == "completed"
    assert saved_updated["metrics"]["faithfulness"] == 0.9
    assert saved_updated["metrics"]["answer_relevancy"] == 0.8
    assert saved_updated["metrics"]["context_precision"] == 0.7
    assert saved_updated["evaluation_id"] == update_record["evaluation_id"]

    print("Evaluation update test passed!")
    print(f"Logging test file: {test_log}")
    print(f"Update test file: {update_log}")


if __name__ == "__main__":
    main()
