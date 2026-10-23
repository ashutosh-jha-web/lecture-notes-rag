
from functools import lru_cache

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from app.config import RETRIEVAL_K, RERANK_TOP_N, RERANKER_MODEL


@lru_cache(maxsize=1)
def get_reranker_model():
    """Load the cross-encoder once and reuse it across questions."""
    return HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL
    )


def make_reranked_retriever(vector_store):
    """Retrieve candidates from FAISS and rerank them by relevance."""

    if RETRIEVAL_K < 1:
        raise ValueError("RETRIEVAL_K must be at least 1.")

    if RERANK_TOP_N < 1:
        raise ValueError("RERANK_TOP_N must be at least 1.")

    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    compressor = CrossEncoderReranker(
        model=get_reranker_model(),
        top_n=RERANK_TOP_N,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
