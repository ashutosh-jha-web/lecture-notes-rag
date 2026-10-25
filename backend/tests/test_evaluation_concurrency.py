import json
from concurrent.futures import ThreadPoolExecutor

from app.evaluation import append_evaluation


def test_concurrent_appends_preserve_all_unique_records(tmp_path):
    log_path = tmp_path / "evaluations.jsonl"
    total_records = 50

    def append_one(index):
        return append_evaluation(
            {
                "evaluation_id": f"test-evaluation-{index}",
                "question": f"Question {index}",
                "status": "pending",
            },
            path=log_path,
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(append_one, range(total_records)))

    with log_path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    ids = [row["evaluation_id"] for row in rows]

    assert len(results) == total_records
    assert len(rows) == total_records
    assert len(set(ids)) == total_records
    assert set(ids) == {
        f"test-evaluation-{index}" for index in range(total_records)
    }
