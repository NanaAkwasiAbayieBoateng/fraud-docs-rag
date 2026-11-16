import pytest

from rag_app.index import Chunk, HybridIndex
from rag_app.ollama_client import FakeOllamaClient

CORPUS = [
    ("scams.md", "Push payment scams trick victims into authorizing bank transfers to criminals."),
    ("mules.md", "Money mules move stolen money through their own accounts for a commission."),
    ("cards.md", "Card-not-present fraud uses stolen card numbers for online purchases."),
]


@pytest.fixture()
def index_and_client():
    client = FakeOllamaClient()
    chunks = [Chunk(text=text, source=source, chunk_index=0) for source, text in CORPUS]
    embeddings = client.embed([c.text for c in chunks])
    return HybridIndex(chunks=chunks, embeddings=embeddings), client


def search(index, client, query, k=3):
    return index.search(query, client.embed_one(query), k=k)


def test_relevant_document_ranks_first(index_and_client):
    index, client = index_and_client
    results = search(index, client, "money mules moving stolen money through accounts")
    assert results[0].chunk.source == "mules.md"


def test_each_topic_finds_its_document(index_and_client):
    index, client = index_and_client
    expectations = {
        "authorized push payment bank transfer scam": "scams.md",
        "stolen card numbers used for online purchases": "cards.md",
    }
    for query, expected in expectations.items():
        results = search(index, client, query)
        assert results[0].chunk.source == expected, query


def test_results_sorted_by_score_descending(index_and_client):
    index, client = index_and_client
    results = search(index, client, "fraud money bank card")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_no_results():
    index = HybridIndex()
    assert index.search("anything", [0.0] * 4, k=3) == []


def test_save_and_load_roundtrip(index_and_client, tmp_path):
    index, client = index_and_client
    path = tmp_path / "index.json"
    index.save(path)
    loaded = HybridIndex.load(path)
    assert len(loaded) == len(index)
    assert [c.text for c in loaded.chunks] == [c.text for c in index.chunks]
    results = search(loaded, client, "money mules stolen money accounts")
    assert results[0].chunk.source == "mules.md"
