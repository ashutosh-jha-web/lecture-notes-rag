
from pathlib import Path

from app.ingestion import extract_pdf_pages, chunk_pages
from app.vector_store import build_and_save_index


def main():
    pdf_path = Path("backend/data/uploads/sample.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Sample PDF not found: {pdf_path.resolve()}"
        )

    print("Extracting PDF pages...")
    pages = extract_pdf_pages(pdf_path)
    print("Pages:", len(pages))

    print("Creating chunks...")
    chunks = chunk_pages(pages)
    print("Chunks:", len(chunks))

    print("Building FAISS index...")
    store = build_and_save_index(chunks)

    print("Searching for relevant chunks...")
    results = store.similarity_search(
        "What is the main topic?",
        k=3,
    )

    if not results:
        raise AssertionError("No search results returned.")

    print("\nTop result metadata:")
    print(results[0].metadata)

    print("\nTop result excerpt:")
    print(results[0].page_content[:300])

    print("\nVector-store test completed successfully!")


if __name__ == "__main__":
    main()
