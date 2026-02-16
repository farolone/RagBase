from rag.generation.citation import CitationGenerator
from rag.storage.qdrant import SearchResult


def test_build_prompt():
    gen = CitationGenerator()
    results = [
        SearchResult(chunk_id="c1", document_id="d1", content="Berlin is the capital.", score=0.9, metadata={"source_url": "https://example.com"}),
        SearchResult(chunk_id="c2", document_id="d2", content="Germany has 83 million people.", score=0.8, metadata={"platform": "web"}),
    ]

    prompt, citation_map = gen.build_prompt("Tell me about Germany", results)

    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Berlin" in prompt
    assert 1 in citation_map
    assert 2 in citation_map


def test_parse_citations():
    gen = CitationGenerator()
    citation_map = {
        1: SearchResult(chunk_id="c1", document_id="d1", content="Berlin info", score=0.9, metadata={"source_url": "https://example.com"}),
        2: SearchResult(chunk_id="c2", document_id="d2", content="Germany info", score=0.8, metadata={}),
    }

    answer = "Berlin is the capital of Germany [1]. The country has 83 million inhabitants [2]."
    result = gen.parse_citations(answer, citation_map)

    assert result["citation_count"] == 2
    assert len(result["sources"]) == 2
    assert result["sources"][0]["ref"] == 1
