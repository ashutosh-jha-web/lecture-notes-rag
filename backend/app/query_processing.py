
import re


def normalize_question(question: str) -> str:
    """Normalize whitespace for consistent query routing."""
    return re.sub(r"\s+", " ", question).strip()


def route_query(question: str) -> str:
    """
    Route a question using rule-based heuristics.

    Returns:
        empty: The question is blank.
        greeting: The question is a simple greeting.
        decompose: The question contains multiple tasks or comparisons.
        retrieve: A normal information-seeking question.
    """
    q = normalize_question(question).lower()

    if not q:
        return "empty"

    # Handle simple greetings.
    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
    }

    if q in greetings:
        return "greeting"

    # Indicators of comparisons or multiple tasks.
    multi_part_patterns = (
        r"\bcompare\b",
        r"\bcontrast\b",
        r"\bdifference between\b",
        r"\bdifferences between\b",
        r"\bpros and cons\b",
        r"\bsimilarities and differences\b",
        r"\band also\b",
        r"\bas well as\b",
        r"\band how\b",
        r"\band what\b",
        r"\band why\b",
        r"\band when\b",
        r"\band where\b",
        r"\bfirst\b.*\bthen\b",
        r"\bon the other hand\b",
    )

    if any(re.search(pattern, q) for pattern in multi_part_patterns):
        return "decompose"

    # Multiple explicit questions.
    if len(re.findall(r"\?", question)) > 1:
        return "decompose"

    # Multiple task verbs in the same question.
    task_verbs = (
        "explain",
        "describe",
        "compare",
        "contrast",
        "define",
        "discuss",
        "identify",
        "list",
        "summarize",
        "evaluate",
    )

    task_verb_pattern = (
        r"\b(" + "|".join(task_verbs) + r")\b"
    )

    if len(re.findall(task_verb_pattern, q)) >= 2:
        return "decompose"

    return "retrieve"


def decompose_query(question: str) -> list[str]:
    """
    Split a multi-part question into distinct subquestions.

    Uses deterministic text heuristics only; it does not call an LLM.
    Preserves the original question if a reliable split cannot be found.
    """
    original = normalize_question(question)

    if not original:
        return []

    # Step 1: Split on question marks, semicolons, and line breaks.
    pieces = re.split(r"\?+|;\s*|\n+", original)

    parts = []

    for piece in pieces:
        piece = piece.strip(" \t\r\n.,;?!")

        if piece:
            parts.append(piece)

    if len(parts) > 1:
        return _clean_subquestions(parts, original)

    # Step 2: Split on task connectors.
    #
    # Examples:
    # "Explain CNNs, and compare them with RNNs."
    # -> "Explain CNNs" + "compare them with RNNs"
    #
    # "What is RAG and how does reranking improve retrieval?"
    # -> "What is RAG" + "how does reranking improve retrieval"

    connector_pattern = re.compile(
        r"\s*,?\s+and\s+"
        r"(?=(?:explain|describe|compare|contrast|define|discuss|"
        r"identify|list|summarize|evaluate|show|give|outline|"
        r"how|why|what|which|when|where)\b)",
        flags=re.IGNORECASE,
    )

    parts = connector_pattern.split(original)

    # Step 3: Handle additional common connectors.
    if len(parts) == 1:
        comparison_pattern = re.compile(
            r"\s+\b(?:and also|as well as)\b\s+",
            flags=re.IGNORECASE,
        )

        parts = comparison_pattern.split(original)

    # Step 4: If no reliable split was found, keep the full query.
    if len(parts) == 1:
        return [original]

    return _clean_subquestions(parts, original)


def _clean_subquestions(
    parts: list[str],
    original: str,
) -> list[str]:
    """Clean subquestions, remove duplicates, and avoid tiny fragments."""
    cleaned = []
    seen = set()

    for part in parts:
        text = part.strip(" \t\r\n.,;?!")
        text = re.sub(r"\s+", " ", text)

        # Avoid fragments that are too short for useful retrieval.
        if len(text.split()) < 2:
            continue

        key = text.casefold()

        if key not in seen:
            cleaned.append(text)
            seen.add(key)

    # Fall back to the original question if splitting was unreliable.
    if len(cleaned) < 2:
        return [original]

    return cleaned
