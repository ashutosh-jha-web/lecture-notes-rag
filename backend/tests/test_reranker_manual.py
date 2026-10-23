
from app.vector_store import load_index
from app.retrieval import make_reranked_retriever


def main():
    print("Loading saved FAISS index...")
    store = load_index()

    print("Creating reranked retriever...")
    retriever = make_reranked_retriever(store)

    print("Searching and reranking chunks...")
    docs = retriever.invoke("Summarize the core concept")

    assert docs, "No documents returned by the reranker."

    print(f"\nRetrieved {len(docs)} reranked chunks:")

    for doc in docs:
        print(
            f"Page: {doc.metadata.get('page')}, "
            f"Chunk: {doc.metadata.get('chunk_id')}"
        )
        print(doc.page_content[:200])
        print("-" * 50)

    print("Reranker test completed successfully!")


if __name__ == "__main__":
    main()
