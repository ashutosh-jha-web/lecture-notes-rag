
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import BACKEND_DIR, EMBEDDING_MODEL, FAISS_DIR


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the local sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def resolve_index_dir(index_dir: str | Path = FAISS_DIR) -> Path:
    """Resolve relative index paths from the backend directory."""
    path = Path(index_dir)

    if not path.is_absolute():
        path = BACKEND_DIR / path

    return path.resolve()


def build_and_save_index(
    chunks: list[Document],
    index_dir: str | Path = FAISS_DIR,
) -> FAISS:
    """Build and save one FAISS index from chunks across multiple PDFs."""
    if not chunks:
        raise ValueError("No chunks supplied. Upload a text-based PDF first.")

    store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )

    target_dir = resolve_index_dir(index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    store.save_local(str(target_dir))

    return store


def load_index(index_dir: str | Path = FAISS_DIR) -> FAISS:
    """Load a previously saved FAISS index created by this application."""
    target_dir = resolve_index_dir(index_dir)
    index_file = target_dir / "index.faiss"
    metadata_file = target_dir / "index.pkl"

    if not index_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"Complete FAISS index not found at {target_dir}. "
            "Upload your PDF files to build an index first."
        )

    # Only load index files that were created by your application
    # and that you trust. The pickle file uses deserialization.
    return FAISS.load_local(
        str(target_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
