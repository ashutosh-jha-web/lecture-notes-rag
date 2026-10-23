
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import CHAT_MODEL, GOOGLE_API_KEY


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the latest user question as a standalone question
for searching lecture notes.

Use conversation history to resolve references such as:
"explain more", "give an example", "why?", "compare them",
"what about the second one", and similar follow-ups.

Preserve the user's intent. If the question is already self-contained,
keep its meaning unchanged. Return only the rewritten question.
Do not answer the question.""",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\n"
            "Latest question:\n{question}",
        ),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer using the supplied lecture context.

Use conversation history only to understand what the user means.
The lecture context is the factual source for your answer.

If the user asks for more detail, explain the concept more thoroughly
and provide examples when supported by the context. Avoid merely
repeating the previous answer.

If the context does not contain enough information, say you cannot
determine it from the provided document. Do not invent facts,
citations, page numbers, or source details.

Be clear and appropriately detailed.""",
        ),
        (
            "human",
            "Conversation history:\n{history}\n\n"
            "Current question: {question}\n\n"
            "Retrieved lecture context:\n{context}",
        ),
    ]
)


def create_llm():
    """Create the configured Gemini model."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY in backend/.env")

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def format_history(history: list[dict] | None) -> str:
    """Format recent user and assistant messages for the model."""
    if not history:
        return "No previous conversation."

    lines = []

    for message in history[-10:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content.strip()[:4000]}")

    return "\n".join(lines) if lines else "No previous conversation."


def rewrite_question(
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Resolve follow-up references before retrieving PDF chunks."""
    if not history:
        return question.strip()

    clean_history = format_history(history)

    # Avoid spending an extra model call on obviously standalone questions.
    normalized = question.lower().strip()
    follow_up_phrases = (
        "explain more",
        "explain it more",
        "explain further",
        "elaborate",
        "elaborate more",
        "tell me more",
        "more details",
        "in more detail",
        "give an example",
        "give me an example",
        "another example",
        "why",
        "why?",
        "how so",
        "what about it",
        "what about them",
        "compare them",
        "compare these",
        "the first one",
        "the second one",
        "could you explain",
        "can you explain",
        "what do you mean",
        "simplify it",
        "make it simpler",
        "explain lil more",
        "explain a little more",
    )

    is_short_follow_up = (
        len(normalized.split()) <= 8
        or any(phrase in normalized for phrase in follow_up_phrases)
    )

    if not is_short_follow_up:
        return question.strip()

    chain = REWRITE_PROMPT | create_llm() | StrOutputParser()

    rewritten = chain.invoke(
        {
            "history": clean_history,
            "question": question.strip(),
        }
    ).strip()

    # Guard against an empty model response.
    return rewritten or question.strip()


def format_docs(docs):
    """Format retrieved chunks with source metadata for Gemini."""
    blocks = []

    for doc in docs:
        metadata = doc.metadata

        tag = (
            f"[source: {metadata.get('source', 'Unknown')}, "
            f"page: {metadata.get('page', 'Unknown')}, "
            f"chunk: {metadata.get('chunk_id', 'Unknown')}]"
        )

        blocks.append(f"{tag}\n{doc.page_content}")

    return "\n\n---\n\n".join(blocks)


def extract_citations(docs) -> list[dict]:
    """Build citations from retrieved metadata, not generated answer text."""
    citations = []

    for doc in docs:
        metadata = doc.metadata

        citations.append(
            {
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "excerpt": doc.page_content[:300],
            }
        )

    return citations


def create_answer_chain():
    """Create the answer-generation chain."""
    return ANSWER_PROMPT | create_llm() | StrOutputParser()


def answer_question(
    question: str,
    docs: list,
    history: list[dict] | None = None,
):
    """Generate a complete answer and return source citations separately."""
    if not docs:
        return {
            "answer": "I cannot determine the answer from the provided document.",
            "sources": [],
        }

    chain = create_answer_chain()

    answer = chain.invoke(
        {
            "question": question,
            "history": format_history(history),
            "context": format_docs(docs),
        }
    )

    return {
        "answer": answer,
        "sources": extract_citations(docs),
    }


def stream_answer(
    question: str,
    docs: list,
    history: list[dict] | None = None,
):
    """Yield the answer progressively for the streaming endpoint."""
    if not docs:
        yield "I cannot determine the answer from the provided document."
        return

    chain = create_answer_chain()

    for chunk in chain.stream(
        {
            "question": question,
            "history": format_history(history),
            "context": format_docs(docs),
        }
    ):
        if chunk:
            yield chunk
