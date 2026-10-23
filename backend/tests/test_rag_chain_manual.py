
from app.vector_store import load_index
from app.retrieval import make_reranked_retriever
from app.rag_chain import answer_question


def main():
    store = load_index()
    retriever = make_reranked_retriever(store)

    question = "What is machine learning?"
    docs = retriever.invoke(question)

    result = answer_question(question, docs)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"Page: {source['page']}, "
            f"Chunk: {source['chunk_id']}, "
            f"Source: {source['source']}"
        )

    assert result["answer"].strip()
    assert result["sources"]

    print("\nRAG answer-generation test passed!")


if __name__ == "__main__":
    main()
