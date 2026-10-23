
from app.query_processing import route_query, decompose_query


def test_route_query():
    assert route_query("") == "empty"
    assert route_query("  ") == "empty"
    assert route_query("Hello") == "greeting"
    assert route_query("What is machine learning?") == "retrieve"
    assert route_query(
        "What is machine learning? What is overfitting?"
    ) == "decompose"
    assert route_query(
        "Compare classification and regression"
    ) == "decompose"


def test_decompose_query():
    result = decompose_query(
        "What is classification? What is regression?"
    )
    assert result == [
        "What is classification",
        "What is regression",
    ]

    result = decompose_query(
        "Explain embeddings; explain vector search"
    )
    assert result == [
        "Explain embeddings",
        "explain vector search",
    ]


def test_single_question_stays_as_one_item():
    result = decompose_query("What is machine learning?")
    assert result == ["What is machine learning?"]
