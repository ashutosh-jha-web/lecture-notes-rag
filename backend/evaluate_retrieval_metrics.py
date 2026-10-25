from pathlib import Path

from app.config import EMBEDDING_MODEL, RETRIEVAL_K, RERANK_TOP_N
from app.vector_store import load_index
from app.retrieval import get_reranker_model


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_PATH = PROJECT_ROOT / "retrieval_metrics_report.txt"

# This benchmark is designed for the currently indexed
# NLP_Detailed_Notes.pdf. Questions outside that document's scope
# are intentionally excluded.
QUESTIONS = [
    "Summarize what natural language processing is",
    "What is TF-IDF and how does it represent text?",
    "Explain tokenization and subword tokenization",
    "How do Word2Vec and GloVe represent words?",
    "How do static and contextual embeddings differ?",
    "What is the role of transformers in NLP?",
    "Explain the RAG workflow",
    "How does a cross-encoder reranker work?",
    "How can retrieval quality be evaluated?",
    "How should a RAG system handle questions unsupported by its documents?",
]

# Candidate relevance labels based on inspection of the actual
# indexed chunk excerpts. Review them against the full PDF before
# treating metrics as a gold-standard benchmark.
RELEVANT_CHUNKS = {
    "Summarize what natural language processing is": {"CHK-2"},
    "What is TF-IDF and how does it represent text?": {"CHK-6"},
    "Explain tokenization and subword tokenization": {"CHK-4", "CHK-5"},
    "How do Word2Vec and GloVe represent words?": {"CHK-7"},
    "How do static and contextual embeddings differ?": {"CHK-7", "CHK-9"},
    "What is the role of transformers in NLP?": {"CHK-9"},
    "Explain the RAG workflow": {"CHK-11"},
    "How does a cross-encoder reranker work?": {"CHK-11", "CHK-12"},
    "How can retrieval quality be evaluated?": {"CHK-12"},
    "How should a RAG system handle questions unsupported by its documents?": {
        "CHK-12",
        "CHK-13",
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


def get_index_documents(vector_store):
    """Return documents stored in the FAISS docstore."""
    documents = []
    mapping = getattr(vector_store, "index_to_docstore_id", {})

    for docstore_id in mapping.values():
        document = vector_store.docstore.search(docstore_id)
        if document is None or isinstance(document, str):
            continue
        documents.append(document)

    return documents


# =========================================================
# METRIC FUNCTIONS
# =========================================================

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Calculate Precision@K using the requested cutoff."""
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for chunk_id in top_k if chunk_id in relevant_ids)
    # Standard Precision@K uses k as denominator, even when fewer than k
    # documents are returned.
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
    documents = vector_store.similarity_search(question, k=k)
    return deduplicate_documents(documents)


def retrieve_reranked(vector_store, question, k, top_n, reranker):
    """Retrieve candidates and rerank them with a cross-encoder."""
    candidates = retrieve_baseline(vector_store, question, k)

    if not candidates:
        return []

    pairs = [(question, document.page_content) for document in candidates]
    scores = reranker.score(pairs)

    if len(scores) != len(candidates):
        raise RuntimeError(
            "Reranker returned a different number of scores than candidates: "
            f"{len(scores)} scores for {len(candidates)} documents."
        )

    ranked_results = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [document for document, _score in ranked_results[:top_n]]


# =========================================================
# RELEVANCE LABEL VALIDATION
# =========================================================

def validate_relevance_labels(vector_store):
    """
    Check that every label refers to an indexed chunk.

    This validates ID existence only; it cannot determine semantic relevance.
    Raises ValueError instead of producing misleading zero scores if labels
    are missing from the active index.
    """
    stored_chunk_ids = {
        get_chunk_id(document)
        for document in get_index_documents(vector_store)
        if get_chunk_id(document) is not None
    }

    print("\n" + "=" * 80)
    print("RELEVANCE LABEL VALIDATION")
    print("=" * 80)
    print(f"Unique chunk IDs found: {len(stored_chunk_ids)}")

    all_labels = set()
    for question in QUESTIONS:
        labeled_ids = RELEVANT_CHUNKS.get(question)
        if not labeled_ids:
            raise ValueError(
                f"No relevance labels configured for question: {question}"
            )
        all_labels.update(labeled_ids)

    missing_ids = all_labels - stored_chunk_ids
    if missing_ids:
        raise ValueError(
            "Evaluation stopped because relevance labels do not match the "
            "currently loaded FAISS index.\n"
            f"Missing chunk IDs: {sorted(missing_ids)}\n"
            "Review the index and RELEVANT_CHUNKS before rerunning."
        )

    for question in QUESTIONS:
        print(
            f"OK: {question} -> "
            f"{sorted(RELEVANT_CHUNKS[question])}"
        )

    print(
        "All labeled IDs exist. Semantic relevance still requires review "
        "against the full source text."
    )
    return stored_chunk_ids


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
    if RERANK_TOP_N > RETRIEVAL_K:
        log(
            "WARNING: RERANK_TOP_N is greater than RETRIEVAL_K; "
            "reranked results cannot exceed the retrieved candidate count."
        )

    vector_store = load_index()
    log("FAISS index loaded successfully.")

    # Fail fast before calculating any metrics if labels are stale.
    validate_relevance_labels(vector_store)

    log("\nLoading cross-encoder reranker...")
    reranker = get_reranker_model()
    log("Cross-encoder loaded successfully.")

    baseline_precisions = []
    reranked_precisions = []
    baseline_rrs = []
    reranked_rrs = []

    for question_number, question in enumerate(QUESTIONS, start=1):
        relevant_ids = RELEVANT_CHUNKS[question]

        log("\n" + "#" * 80)
        log(f"QUESTION {question_number}: {question}")
        log("#" * 80)
        log(f"Manually labeled relevant IDs: {sorted(relevant_ids)}")

        baseline_documents = retrieve_baseline(
            vector_store, question, RETRIEVAL_K
        )
        reranked_documents = retrieve_reranked(
            vector_store,
            question,
            RETRIEVAL_K,
            RERANK_TOP_N,
            reranker,
        )

        baseline_ids = [get_chunk_id(document) for document in baseline_documents]
        reranked_ids = [get_chunk_id(document) for document in reranked_documents]

        log("\nBASELINE FAISS RESULTS")
        if not baseline_documents:
            log("No documents retrieved.")
        for rank, document in enumerate(baseline_documents, start=1):
            chunk_id = get_chunk_id(document)
            label = "LABELED RELEVANT" if chunk_id in relevant_ids else "NOT LABELED RELEVANT"
            log(f"\nRank: {rank}")
            log(f"Chunk ID: {chunk_id}")
            log(f"Label: {label}")
            log(f"Source: {get_source_info(document)}")
            log(f"Excerpt: {get_excerpt(document)}")

        log("\nCROSS-ENCODER RERANKED RESULTS")
        if not reranked_documents:
            log("No documents retrieved.")
        for rank, document in enumerate(reranked_documents, start=1):
            chunk_id = get_chunk_id(document)
            label = "LABELED RELEVANT" if chunk_id in relevant_ids else "NOT LABELED RELEVANT"
            log(f"\nRank: {rank}")
            log(f"Chunk ID: {chunk_id}")
            log(f"Label: {label}")
            log(f"Source: {get_source_info(document)}")
            log(f"Excerpt: {get_excerpt(document)}")

        baseline_p = precision_at_k(baseline_ids, relevant_ids, RERANK_TOP_N)
        reranked_p = precision_at_k(reranked_ids, relevant_ids, RERANK_TOP_N)
        baseline_r = reciprocal_rank(
            baseline_ids[:RERANK_TOP_N], relevant_ids
        )
        reranked_r = reciprocal_rank(
            reranked_ids[:RERANK_TOP_N], relevant_ids
        )
        baseline_precisions.append(baseline_p)
        reranked_precisions.append(reranked_p)
        baseline_rrs.append(baseline_r)
        reranked_rrs.append(reranked_r)

        log("\nQUESTION METRICS")
        log(f"Baseline Precision@{RERANK_TOP_N}: {baseline_p:.4f}")
        log(f"Reranked Precision@{RERANK_TOP_N}: {reranked_p:.4f}")
        log(f"Baseline Reciprocal Rank: {baseline_r:.4f}")
        log(f"Reranked Reciprocal Rank: {reranked_r:.4f}")

    log("\n" + "=" * 80)
    log("AGGREGATE RESULTS")
    log("=" * 80)
    log(f"Evaluated questions: {len(QUESTIONS)}")
    log("Skipped questions: 0")

    baseline_mean_p = sum(baseline_precisions) / len(baseline_precisions)
    reranked_mean_p = sum(reranked_precisions) / len(reranked_precisions)
    baseline_mrr = sum(baseline_rrs) / len(baseline_rrs)
    reranked_mrr = sum(reranked_rrs) / len(reranked_rrs)

    log(f"Baseline Mean Precision@{RERANK_TOP_N}: {baseline_mean_p:.4f}")
    log(f"Reranked Mean Precision@{RERANK_TOP_N}: {reranked_mean_p:.4f}")
    log(f"Precision change: {reranked_mean_p - baseline_mean_p:+.4f}")
    log(f"Baseline MRR: {baseline_mrr:.4f}")
    log(f"Reranked MRR: {reranked_mrr:.4f}")
    log(f"MRR change: {reranked_mrr - baseline_mrr:+.4f}")

    log("\n" + "=" * 80)
    log("INTERPRETATION NOTES")
    log("=" * 80)
    log(
        "Scores depend on the correctness and completeness of manual labels. "
        "The labels in this script are candidate labels based on inspected "
        "chunk excerpts and should be reviewed against the full source PDF."
    )
    log(
        "This benchmark evaluates only the currently indexed NLP notes. "
        "It does not establish performance on CNN, general deep-learning, "
        "or machine-learning documents that are not in the index."
    )
    log(
        "A reranker can reorder only the candidates returned by the initial "
        "retriever; it cannot recover evidence that was not retrieved."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
