
from app.ragas_evaluator import evaluate_rag_response


question = "What is machine learning?"

answer = (
    "Machine learning is a branch of artificial intelligence "
    "that enables systems to learn patterns from data."
)

contexts = [
    (
        "Machine learning is a subset of artificial intelligence "
        "that allows computers to learn from data and identify patterns "
        "without being explicitly programmed for every task."
    ),
    (
        "Supervised learning uses labeled data, while unsupervised "
        "learning discovers patterns in unlabeled data."
    ),
]

print("Starting RAGAS evaluation...")
print("This may take a little while because evaluation uses model calls.")

try:
    scores = evaluate_rag_response(
        question=question,
        answer=answer,
        contexts=contexts,
    )

    print("\nRAGAS Evaluation Results")
    print("-" * 35)

    for metric_name, score in scores.items():
        if score is None:
            print(f"{metric_name}: Failed to calculate")
        else:
            print(f"{metric_name}: {score:.4f}")

    if all(score is None for score in scores.values()):
        print("\nEvaluation failed: no metrics were calculated.")
    elif any(score is None for score in scores.values()):
        print("\nEvaluation partially completed.")
    else:
        print("\nAll metrics calculated successfully.")

except Exception as exc:
    print("\nEvaluation failed.")
    print(f"{type(exc).__name__}: {exc}")
    raise
