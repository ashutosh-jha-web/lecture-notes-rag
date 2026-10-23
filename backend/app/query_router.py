
import re


def normalize_question(question: str) -> str:
    question = question.lower().strip()
    question = re.sub(r"[^\w\s']", "", question)
    question = re.sub(r"\s+", " ", question)
    return question


def is_greeting(question: str) -> bool:
    normalized = normalize_question(question)

    # Exact greetings
    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "how are you",
        "how are you doing",
        "how are you today",
        "hello how are you",
        "hi how are you",
        "hey how are you",
        "good morning",
        "good afternoon",
        "good evening",
        "what's up",
        "whats up",
        "thank you",
        "thanks",
        "bye",
        "goodbye",
    }

    if normalized in greetings:
        return True

    # Greetings followed by a common conversational phrase
    if re.fullmatch(
        r"(hi|hello|hey)( there)?[, ]+(how are you( doing| today)?)",
        normalized,
    ):
        return True

    return False


def greeting_response(question: str) -> str:
    normalized = normalize_question(question)

    if re.search(r"\bhow are you\b", normalized):
        return (
            "I'm doing well, thanks for asking! "
            "How can I help you with your lecture notes today?"
        )

    if normalized in {"thank you", "thanks"}:
        return "You're welcome! Feel free to ask me anything about your lecture notes."

    if normalized in {"bye", "goodbye"}:
        return "Goodbye! Have a great day."

    if normalized.startswith("good morning"):
        return "Good morning! What would you like to learn today?"

    if normalized.startswith("good afternoon"):
        return "Good afternoon! How can I help?"

    if normalized.startswith("good evening"):
        return "Good evening! What can I help you with?"

    return "Hello! How can I help you with your lecture notes today?"
