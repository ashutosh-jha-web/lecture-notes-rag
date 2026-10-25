from app.vector_store import load_index
from app.retrieval import make_reranked_retriever
from app.rag_chain import answer_question


ABSTENTION_PHRASES = (
    "cannot determine",
    "can't determine",
    "not provided",
    "not specified",
    "does not contain",
    "do not contain",
    "insufficient information",
    "not enough information",
    "cannot find",
    "can't find",
)


def is_abstention(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def run_case(retriever, name, question, expect_abstention=False):
    print(f"\n{'=' * 70}")
    print(f"CASE: {name}")
    print(f"QUESTION: {question}")

    docs = retriever.invoke(question)
    result = answer_question(question, docs)
    answer = result["answer"].strip()
    sources = result["sources"]

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    if not sources:
        print("No sources returned.")
    for source in sources:
        print(
            f"Page: {source['page']}, "
            f"Chunk: {source['chunk_id']}, "
            f"Source: {source['source']}"
        )

    if not answer:
        passed = False
        reason = "Empty answer"
    elif expect_abstention:
        passed = is_abstention(answer)
        reason = (
            "Explicit abstention detected"
            if passed
            else "Expected an explicit abstention, but none was detected"
        )
    else:
        passed = bool(sources)
        reason = (
            "Non-empty answer with source metadata"
            if passed
            else "Expected source metadata, but no sources were returned"
        )

    print(f"\nRESULT: {'PASS' if passed else 'FAIL'} — {reason}")
    return passed


def main():
    store = load_index()
    retriever = make_reranked_retriever(store)

    cases = [
        (
            "Answerable question",
            "What is TF-IDF?",
            False,
        ),
        (
            "Out-of-scope question",
            "What is the capital of France?",
            True,
        ),
        (
            "Unsupported exact detail",
            "What is the exact TF-IDF score of document 17?",
            True,
        ),
    ]

    results = []
    for name, question, expect_abstention in cases:
        results.append(
            (
                name,
                run_case(
                    retriever,
                    name,
                    question,
                    expect_abstention=expect_abstention,
                ),
            )
        )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    for name, passed in results:
        print(f"{'PASS' if passed else 'FAIL'} — {name}")

    passed_count = sum(passed for _, passed in results)
    print(f"\nPassed: {passed_count}/{len(results)}")

    if passed_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
