
from pathlib import Path
from datetime import datetime

from app.config import EMBEDDING_MODEL, RETRIEVAL_K, RERANK_TOP_N
from app.vector_store import load_index
from app.retrieval import get_reranker_model


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "retrieval_metrics_report.txt"

QUESTIONS = [
    "Summarize what is NLP",
    "What is TF-IDF and how does it represent text?",
    "Explain how convolutional neural networks extract features from images.",
    "What is the difference between machine learning and deep learning?",
    "What are word embeddings in NLP?",
    "Explain tokenization in natural language processing.",
    "What is the role of transformers in NLP?",
    "What is overfitting in machine learning?",
    "Explain the role of activation functions in neural networks.",
    "What is backpropagation in deep learning?",
]

# These are your existing starting labels.
# Verify them against the actual chunk contents before
# treating the final metrics as ground truth.

RELEVANT_CHUNKS = {
    "Summarize what is NLP": {
        "45f34d1ecb-P2-C0",
        "45f34d1ecb-P2-C663",
    },
    "What is TF-IDF and how does it represent text?": {
        "45f34d1ecb-P2-C2883",
    },
    "Explain how convolutional neural networks extract features from images.": {
        "319683de7a-P3-C736",
    },
    "What is the difference between machine learning and deep learning?": {
        "ac0cc75022-P2-C0",
        "319683de7a-P2-C0",
    },
    "What are word embeddings in NLP?": {
        "45f34d1ecb-P3-C0",
    },
    "Explain tokenization in natural language processing.": {
        "45f34d1ecb-P2-C2072",
    },
    "What is the role of transformers in NLP?": {
        "319683de7a-P3-C1490",
        "319683de7a-P3-C2256",
    },
    "What is overfitting in machine learning?": {
        "319683de7a-P3-C0",
        "ac0cc75022-P4-C725",
    },
    "Explain the role of activation functions in neural networks.": {
        "319683de7a-P2-C0",
    },
    "What is backpropagation in deep learning?": {
        "319683de7a-P2-C713",
    },
}


# =========================================================
# DOCUMENT HELPERS
# =========================================================

def get_chunk_id(document):
    """Get the chunk ID stored in document metadata."""
    metadata = document.metadata or {}

    for key in ("chunk_id", "id", "chunk"):
        value = metadata.get(key)
        if value is not None:
            return str(value)

    return None


def get_source_info(document):
    """Get source filename, page number and chunk ID."""
    metadata = document.metadata or {}

    source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("file_name")
        or "Unknown source"
    )

    page = metadata.get("page", metadata.get("page_number", "?"))
    chunk_id = get_chunk_id(document)

    return (
        f"File={Path(str(source)).name}, "
        f"Page={page}, Chunk={chunk_id}"
    )


def get_excerpt(document, max_chars=500):
    """Return a readable excerpt from a document."""
    text = " ".join((document.page_content or "").split())
    return text[:max_chars] or "[Empty document content]"


def deduplicate_documents(documents):
    """Remove duplicate chunk IDs while preserving order."""
    unique_documents = []
    seen_ids = set()

    for document in documents:
        chunk_id = get_chunk_id(document)

        if chunk_id is None:
            print(
                "WARNING: A document has no chunk_id metadata. "
                "Check your PDF ingestion metadata."
            )
            continue

        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_documents.append(document)

    return unique_documents


# =========================================================
# METRIC FUNCTIONS
# =========================================================

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Calculate Precision@K."""
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_count / k


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Calculate reciprocal rank of the first relevant result."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


# =========================================================
# RETRIEVAL FUNCTIONS
# =========================================================

def retrieve_baseline(vector_store, question, k):
    """Retrieve top-k documents using FAISS similarity search."""
    documents = vector_store.similarity_search(
        question,
        k=k,
    )

    return deduplicate_documents(documents)


def retrieve_reranked(
    vector_store,
    question,
    k,
    top_n,
    reranker,
):
    """Retrieve candidates and rerank them with a cross-encoder."""
    candidates = retrieve_baseline(
        vector_store,
        question,
        k,
    )

    if not candidates:
        return []

    pairs = [
        (question, document.page_content)
        for document in candidates
    ]

    scores = reranker.score(pairs)

    ranked_results = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [
        document
        for document, _score in ranked_results[:top_n]
    ]


# =========================================================
# RELEVANCE LABEL VALIDATION
# =========================================================

def validate_relevance_labels(vector_store):
    """
    Check whether manually labeled chunk IDs exist in FAISS.

    This checks ID existence only. It cannot decide whether
    a chunk is semantically relevant to a question.
    """
    stored_chunk_ids = set()
    missing_metadata_count = 0

    mapping = getattr(
        vector_store,
        "index_to_docstore_id",
        {},
    )

    for docstore_id in mapping.values():
        document = vector_store.docstore.search(docstore_id)

        if document is None or isinstance(document, str):
            continue

        chunk_id = get_chunk_id(document)

        if chunk_id is None:
            missing_metadata_count += 1
        else:
            stored_chunk_ids.add(chunk_id)

    print("\n" + "=" * 80)
    print("RELEVANCE LABEL VALIDATION")
    print("=" * 80)
    print(f"Unique chunk IDs found: {len(stored_chunk_ids)}")

    if missing_metadata_count:
        print(
            "WARNING: Documents without chunk IDs: "
            f"{missing_metadata_count}"
        )

    all_labels_exist = True

    for question in QUESTIONS:
        labeled_ids = RELEVANT_CHUNKS.get(question, set())
        missing_ids = labeled_ids - stored_chunk_ids

        if missing_ids:
            all_labels_exist = False
            print(f"\nQuestion: {question}")
            print(f"Missing labeled IDs: {sorted(missing_ids)}")

    if all_labels_exist:
        print("All labeled chunk IDs exist in the FAISS index.")

    print(
        "Note: Existing IDs still need manual semantic relevance "
        "verification against their source text."
    )

    return stored_chunk_ids


# =========================================================
# REPORT HELPERS
# =========================================================

def print_results(title, documents, relevant_ids):
    """Print ranked results and return their chunk IDs."""
    print(f"\n{title}")

    retrieved_ids = []

    if not documents:
        print("No documents retrieved.")
        return retrieved_ids

    for rank, document in enumerate(documents, start=1):
        chunk_id = get_chunk_id(document)
        retrieved_ids.append(chunk_id)

        if chunk_id in relevant_ids:
            label = "LABELED RELEVANT"
        else:
            label = "NOT LABELED RELEVANT"

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Label: {label}")
        print(f"Source: {get_source_info(document)}")
        print(f"Excerpt: {get_excerpt(document)}")

    return retrieved_ids


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():
    report_lines = []

    def log(message=""):
        message = str(message)
        print(message)
        report_lines.append(message)

    log("=" * 80)
    log("RAG RETRIEVAL METRICS EVALUATION")
    log("=" * 80)
    log(f"Embedding model: {EMBEDDING_MODEL}")
    log(f"RETRIEVAL_K: {RETRIEVAL_K}")
    log(f"RERANK_TOP_N: {RERANK_TOP_N}")
    log(f"Questions: {len(QUESTIONS)}")

    if RETRIEVAL_K < 1:
        raise ValueError("RETRIEVAL_K must be at least 1.")

    if RERANK_TOP_N < 1:
        raise ValueError("RERANK_TOP_N must be at least 1.")

    # Load the existing FAISS index using the actual function
    # defined in backend/app/vector_store.py.
    vector_store = load_index()

    log("FAISS index loaded successfully.")

    # Verify whether the labeled IDs exist in the index.
    validate_relevance_labels(vector_store)

    log("\nLoading cross-encoder reranker...")
    reranker = get_reranker_model()
    log("Cross-encoder loaded successfully.")

    baseline_precisions = []
    reranked_precisions = []
    baseline_rrs = []
    reranked_rrs = []

    evaluated_questions = 0
    skipped_questions = 0

    for question_number, question in enumerate(QUESTIONS, start=1):
        relevant_ids = RELEVANT_CHUNKS.get(question, set())

        log("\n" + "#" * 80)
        log(f"QUESTION {question_number}: {question}")
        log("#" * 80)

        if not relevant_ids:
            log("SKIPPED: No relevance labels provided.")
            skipped_questions += 1
            continue

        baseline_documents = retrieve_baseline(
            vector_store,
            question,
            RETRIEVAL_K,
        )

        reranked_documents = retrieve_reranked(
            vector_store,
            question,
            RETRIEVAL_K,
            RERANK_TOP_N,
            reranker,
        )

        # Print results to the terminal and save them to the report.
        # Capture the same details in the report output.
        log("\nBASELINE FAISS RESULTS")
        baseline_ids = []

        for rank, document in enumerate(baseline_documents, start=1):
            chunk_id = get_chunk_id(document)
            baseline_ids.append(chunk_id)

            label = (
                "LABELED RELEVANT"
                if chunk_id in relevant_ids
                else "NOT LABELED RELEVANT"
            )

            log(f"\nRank: {rank}")
            log(f"Chunk ID: {chunk_id}")
            log(f"Label: {label}")
            log(f"Source: {get_source_info(document)}")
            log(f"Excerpt: {get_excerpt(document)}")

        log("\nCROSS-ENCODER RERANKED RESULTS")
        reranked_ids = []

        for rank, document in enumerate(reranked_documents, start=1):
            chunk_id = get_chunk_id(document)
            reranked_ids.append(chunk_id)

            label = (
                "LABELED RELEVANT"
                if chunk_id in relevant_ids
                else "NOT LABELED RELEVANT"
            )

            log(f"\nRank: {rank}")
            log(f"Chunk ID: {chunk_id}")
            log(f"Label: {label}")
            log(f"Source: {get_source_info(document)}")
            log(f"Excerpt: {get_excerpt(document)}")

        baseline_p = precision_at_k(
            baseline_ids,
            relevant_ids,
            RERANK_TOP_N,
        )

        reranked_p = precision_at_k(
            reranked_ids,
            relevant_ids,
            RERANK_TOP_N,
        )

        baseline_r = reciprocal_rank(
            baseline_ids,
            relevant_ids,
        )

        reranked_r = reciprocal_rank(
            reranked_ids,
            relevant_ids,
        )

        baseline_precisions.append(baseline_p)
        reranked_precisions.append(reranked_p)
        baseline_rrs.append(baseline_r)
        reranked_rrs.append(reranked_r)
        evaluated_questions += 1

        log("\nQUESTION METRICS")
        log(f"Relevant chunk IDs: {sorted(relevant_ids)}")
        log(f"Baseline Precision@{RERANK_TOP_N}: {baseline_p:.4f}")
        log(f"Reranked Precision@{RERANK_TOP_N}: {reranked_p:.4f}")
        log(f"Baseline Reciprocal Rank: {baseline_r:.4f}")
        log(f"Reranked Reciprocal Rank: {reranked_r:.4f}")

    log("\n" + "=" * 80)
    log("AGGREGATE RESULTS")
    log("=" * 80)
    log(f"Evaluated questions: {evaluated_questions}")
    log(f"Skipped questions: {skipped_questions}")

    if evaluated_questions:
        baseline_mean_p = (
            sum(baseline_precisions) / evaluated_questions
        )
        reranked_mean_p = (
            sum(reranked_precisions) / evaluated_questions
        )

        baseline_mrr = sum(baseline_rrs) / evaluated_questions
        reranked_mrr = sum(reranked_rrs) / evaluated_questions

        log(
            f"Baseline Mean Precision@{RERANK_TOP_N}: "
            f"{baseline_mean_p:.4f}"
        )
        log(
            f"Reranked Mean Precision@{RERANK_TOP_N}: "
            f"{reranked_mean_p:.4f}"
        )
        log(
            "Precision change: "
            f"{reranked_mean_p - baseline_mean_p:+.4f}"
        )
        log(f"Baseline MRR: {baseline_mrr:.4f}")
        log(f"Reranked MRR: {reranked_mrr:.4f}")
        log(f"MRR change: {reranked_mrr - baseline_mrr:+.4f}")

    log("\n" + "=" * 80)
    log("INTERPRETATION NOTES")
    log("=" * 80)
    log(
        "The metrics depend on the correctness and completeness "
        "of the manual relevance labels."
    )
    log(
        "Read the excerpts and verify them against the source PDFs. "
        "A chunk labeled NOT LABELED RELEVANT may still be relevant "
        "if the manual labels are incomplete."
    )
    log(
        "ID validation only confirms that labeled chunk IDs exist "
        "in FAISS; it does not validate their semantic relevance."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    log(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
