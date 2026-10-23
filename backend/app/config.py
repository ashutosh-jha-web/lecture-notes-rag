
import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the backend directory and load backend/.env
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

# Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")

# Local embedding and reranking models
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# Storage paths
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
FAISS_DIR = Path(os.getenv("FAISS_DIR", "data/faiss_index"))
EVAL_LOG_PATH = Path(os.getenv("EVAL_LOG_PATH", "eval/results.jsonl"))

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "12"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "4"))

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
