
import json
from pathlib import Path

from langchain_core.documents import Document

from app.evaluation import (
    append_evaluation,
    create_evaluation_record,
)


def main():
    test_log = Path(__file__).resolve().parents[1] / "eval" / "test_results.jsonl"

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
        status="success",
    )

    saved = append_evaluation(record, path=test_log)

    with test_log.open("r", encoding="utf-8") as file:
        loaded = json.loads(file.readlines()[-1])

    assert loaded["question"] == "What is machine learning?"
    assert loaded["sources"][0]["page"] == 1
    assert loaded["metrics"]["faithfulness"] is None
    assert loaded["status"] == "success"
    assert loaded["timestamp"]

    print("Evaluation logging test passed!")
    print(f"Test record saved to: {test_log}")
    print(json.dumps(loaded, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
