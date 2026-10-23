
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to answer using the uploaded PDF",
    )


class SourceItem(BaseModel):
    source: str
    page: int
    chunk_id: str
    excerpt: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    message: str


class StatusResponse(BaseModel):
    document_loaded: bool
    filename: str | None = None
    pages: int = 0
    chunks: int = 0
