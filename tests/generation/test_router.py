from rag.generation.router import QueryRouter
from rag.config import settings


def test_route_simple_query():
    router = QueryRouter()
    model = router.route("What is Berlin?")
    assert model == settings.llm_model_rag


def test_route_complex_query():
    router = QueryRouter()
    model = router.route("Compare all sources about machine learning")
    assert model == settings.llm_model_agent


def test_route_code_query():
    router = QueryRouter()
    model = router.route("Write a Python function to parse JSON")
    assert model == settings.llm_model_agent


def test_route_long_context():
    router = QueryRouter()
    model = router.route("Simple question", context_length=200_000)
    assert model == settings.llm_model_agent


def test_route_german_complex():
    router = QueryRouter()
    model = router.route("Vergleiche alle Quellen zum Thema KI")
    assert model == settings.llm_model_agent
