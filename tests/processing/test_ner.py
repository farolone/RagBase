import pytest
from rag.processing.ner import EntityExtractor


@pytest.fixture(scope="module")
def extractor():
    return EntityExtractor()


def test_extract_person(extractor):
    entities = extractor.extract("Angela Merkel visited Berlin yesterday.")
    names = [e.name for e in entities]
    assert "Angela Merkel" in names


def test_extract_location(extractor):
    entities = extractor.extract("Berlin is the capital of Germany.")
    types = {e.name: e.entity_type for e in entities}
    assert types.get("Berlin") == "LOCATION" or types.get("Germany") == "LOCATION"


def test_extract_german(extractor):
    entities = extractor.extract(
        "Die Bundeskanzlerin besuchte das Brandenburger Tor in Berlin."
    )
    names = [e.name for e in entities]
    assert "Berlin" in names or "Brandenburger Tor" in names


def test_extract_returns_entities(extractor):
    entities = extractor.extract(
        "Google was founded by Larry Page and Sergey Brin in Mountain View."
    )
    assert len(entities) >= 2
    types = {e.entity_type for e in entities}
    assert "PERSON" in types or "ORGANIZATION" in types
