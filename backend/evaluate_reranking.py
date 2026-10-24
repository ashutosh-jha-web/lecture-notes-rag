
import json
import sys
from pathlib import Path

# Allow imports from backend/app when run as a script.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))

from app.config import RETRIEVAL_K, RERANK_TOP_N
from app.retrieval import make_reranked_retriever
from app.vector_store import load_index
from reranking_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


QUESTIONS = [
    "What is TF-IDF and how does it represent text?",
    "Explain how convolutional neural networks extract features from images.",
    "What is the difference between machine learning and deep learning?",
]


def chunk_identity(doc):
    """Identify a chunk using its source, page, and chunk ID."""
    metadata = doc.metadata or {}

    return (
        metadata.get("source", "Unknown"),
        metadata.get("page", "Unknown"),
        metadata.get("chunk_id", "Unknown"),
    )


def show_chunk(rank, doc):
    """Print a chunk's rank, metadata, and text excerpt."""
    source, page, chunk_id = chunk_identity(doc)
    excerpt = " ".join((doc.page_content or "").split())[:350]

    print(f"\n  Rank: {rank}")
    print(f"  Source: {source}")
    print(f"  Page: {page}")
    print(f"  Chunk ID: {chunk_id}")
    print(f"  Excerpt: {excerpt}")


def evaluate_question(store, question, relevance_config):
    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)

    # Stage 1: retrieve candidates using FAISS vector similarity.
    base_retriever = store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )
    initial_docs = base_retriever.invoke(question)

    # Stage 2: run the application's FAISS + cross-encoder pipeline.
    reranked_retriever = make_reranked_retriever(store)
    reranked_docs = reranked_retriever.invoke(question)

    relevance = relevance_config[question]["relevant_chunks"]
    initial_ids = [
        "|".join(map(str, chunk_identity(doc)))
        for doc in initial_docs
    ]
    reranked_ids = [
        "|".join(map(str, chunk_identity(doc)))
        for doc in reranked_docs
    ]

    print("\nRELEVANCE METRICS")
    print("-" * 50)
    metrics = {}
    for label, ranked_ids in (
        ("FAISS baseline", initial_ids),
        ("After reranking", reranked_ids),
    ):
        metrics[label] = {
            "Recall@4": recall_at_k(ranked_ids, relevance, 4),
            "MRR": mean_reciprocal_rank(ranked_ids, relevance),
            "nDCG@4": ndcg_at_k(ranked_ids, relevance, 4),
        }
        print(
            f"{label}: "
            + ", ".join(
                f"{name}={value:.3f}"
                for name, value in metrics[label].items()
            )
        )

    # Print the top 5 chunks before reranking.
    print(f"\nTOP 5 BEFORE RERANKING (from {len(initial_docs)} FAISS candidates)")

    for rank, doc in enumerate(initial_docs[:5], start=1):
        show_chunk(rank, doc)

    # Print the final ranking.
    print(f"\nAFTER RERANKING: {len(reranked_docs)} selected chunks")

    for rank, doc in enumerate(reranked_docs, start=1):
        show_chunk(rank, doc)

    # Build rank maps. These preserve each chunk's actual position.
    initial_ranks = {
        chunk_identity(doc): rank
        for rank, doc in enumerate(initial_docs, start=1)
    }

    reranked_ranks = {
        chunk_identity(doc): rank
        for rank, doc in enumerate(reranked_docs, start=1)
    }

    print("\nRANK COMPARISON")
    print("-" * 100)
    print(
        f"{'Before':<12}"
        f"{'After':<12}"
        f"{'Change':<18}"
        f"{'Source / Page / Chunk ID'}"
    )
    print("-" * 100)

    improved = 0
    dropped = 0
    unchanged = 0
    missing_before = 0

    # Compare every chunk returned after reranking with its original rank.
    for identity, after_rank in reranked_ranks.items():
        source, page, chunk_id = identity
        before_rank = initial_ranks.get(identity)

        if before_rank is None:
            before_text = "Not found"
            change = "Not in baseline"
            missing_before += 1

        elif after_rank < before_rank:
            before_text = str(before_rank)
            change = f"Improved +{before_rank - after_rank}"
            improved += 1

        elif after_rank > before_rank:
            before_text = str(before_rank)
            change = f"Dropped -{after_rank - before_rank}"
            dropped += 1

        else:
            before_text = str(before_rank)
            change = "Unchanged"
            unchanged += 1

        print(
            f"{before_text:<12}"
            f"{after_rank:<12}"
            f"{change:<18}"
            f"{source} / page {page} / {chunk_id}"
        )

    # Check whether the relative order of the retained chunks changed.
    initial_retained_order = [
        chunk_identity(doc)
        for doc in initial_docs
        if chunk_identity(doc) in reranked_ranks
    ]

    final_order = [
        chunk_identity(doc)
        for doc in reranked_docs
    ]

    order_changed = initial_retained_order != final_order

    print("\nSUMMARY")
    print("-" * 50)
    print(f"FAISS candidates retrieved: {len(initial_docs)}")
    print(f"Chunks returned after reranking: {len(reranked_docs)}")
    print(f"Chunks improved in rank: {improved}")
    print(f"Chunks dropped in rank: {dropped}")
    print(f"Chunks with unchanged rank: {unchanged}")
    print(f"Chunks not found in initial candidates: {missing_before}")
    print(f"Relative order changed: {order_changed}")

    if not reranked_docs:
        print("WARNING: The reranker returned no chunks.")
    elif order_changed:
        print("RESULT: The order of retained chunks changed.")
    else:
        print("RESULT: The relative order of retained chunks did not change.")

    print(
        "\nMANUAL RELEVANCE CHECK\n"
        "Compare the excerpts with the question. Verify whether the "
        "most relevant chunks appear near the top after reranking. "
        "A changed ranking alone does not prove improved relevance."
    )

    return metrics


def main():
    print("Loading relevance labels...")
    relevance_path = BACKEND_DIR / "reranking_relevance.json"
    with relevance_path.open("r", encoding="utf-8") as file:
        relevance_config = json.load(file)

    missing_questions = set(QUESTIONS) - set(relevance_config)
    if missing_questions:
        raise ValueError(
            f"Missing relevance labels for questions: {sorted(missing_questions)}"
        )

    print("Loading the existing FAISS index...")
    store = load_index()

    report_path = PROJECT_ROOT / "reranking_evaluation.txt"
    original_stdout = sys.stdout

    class Tee:
        """Write output to both the terminal and the report file."""

        def __init__(self, console, report):
            self.console = console
            self.report = report

        def write(self, text):
            self.console.write(text)
            self.report.write(text)

        def flush(self):
            self.console.flush()
            self.report.flush()

    try:
        with report_path.open("w", encoding="utf-8") as report:
            sys.stdout = Tee(original_stdout, report)

            print("RERANKING EVALUATION")
            print(f"RETRIEVAL_K = {RETRIEVAL_K}")
            print(f"RERANK_TOP_N = {RERANK_TOP_N}")
            print(f"Questions evaluated = {len(QUESTIONS)}")

            metric_totals = {
                label: {"Recall@4": 0.0, "MRR": 0.0, "nDCG@4": 0.0}
                for label in ("FAISS baseline", "After reranking")
            }

            for question in QUESTIONS:
                question_metrics = evaluate_question(
                    store, question, relevance_config
                )
                for label, values in question_metrics.items():
                    for metric, value in values.items():
                        metric_totals[label][metric] += value

            print("\nAGGREGATE METRICS (mean across questions)")
            for label, totals in metric_totals.items():
                averages = {
                    metric: value / len(QUESTIONS)
                    for metric, value in totals.items()
                }
                print(
                    f"{label}: "
                    + ", ".join(
                        f"{metric}={value:.3f}"
                        for metric, value in averages.items()
                    )
                )

            print(f"\nReport saved to: {report_path}")

    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
