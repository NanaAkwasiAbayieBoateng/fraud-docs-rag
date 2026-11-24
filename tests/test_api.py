import pytest
from fastapi.testclient import TestClient

from rag_app.index import Chunk, HybridIndex
from rag_app.main import REFUSAL_MESSAGE, create_app
from rag_app.ollama_client import FakeOllamaClient

DOCS = [
    ("app_scams.md",
     "Authorized push payment scams trick victims into authorizing bank "
     "transfers to accounts controlled by criminals impersonating bank staff."),
    ("money_mules.md",
     "Money mules receive stolen money into their bank accounts and move it "
     "onward for criminals, keeping a small commission."),
]


@pytest.fixture()
def client():
    fake = FakeOllamaClient()
    chunks = [Chunk(text=text, source=source, chunk_index=0) for source, text in DOCS]
    index = HybridIndex(chunks=chunks, embeddings=fake.embed([c.text for c in chunks]))
    app = create_app(client=fake, index=index, refusal_threshold=0.25, log_path="")
    return TestClient(app)


def test_health_reports_index(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 2
    assert set(body["sources"]) == {"app_scams.md", "money_mules.md"}


def test_ask_answers_with_citations(client):
    resp = client.post("/ask", json={"question": "What is an authorized push payment scam?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["citations"]
    assert body["citations"][0]["source"] == "app_scams.md"
    assert body["best_score"] >= 0.25


def test_ask_refuses_off_topic_question(client):
    resp = client.post(
        "/ask", json={"question": "How do I bake sourdough bread at home?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["citations"] == []


def test_ask_refuses_with_empty_index():
    fake = FakeOllamaClient()
    app = create_app(client=fake, index=HybridIndex(), refusal_threshold=0.25, log_path="")
    resp = TestClient(app).post("/ask", json={"question": "What is a money mule?"})
    assert resp.json()["refused"] is True


def _two_doc_app(**kwargs):
    fake = FakeOllamaClient()
    chunks = [Chunk(text=text, source=source, chunk_index=0) for source, text in DOCS]
    index = HybridIndex(chunks=chunks, embeddings=fake.embed([c.text for c in chunks]))
    return create_app(client=fake, index=index, refusal_threshold=0.25, **kwargs)


def test_search_returns_ranked_hits_without_answering(client):
    resp = client.post("/search", json={"question": "How are money mules recruited?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"]
    assert body["hits"][0]["source"] == "money_mules.md"
    assert body["hits"][0]["vector_score"] >= body["best_score"] - 1e-9
    assert "answer" not in body


def test_search_source_filter_restricts_results(client):
    resp = client.post(
        "/search",
        json={"question": "How are money mules recruited?", "source": "app_scams.md"},
    )
    hits = resp.json()["hits"]
    assert hits and all(h["source"] == "app_scams.md" for h in hits)


def test_ask_source_filter_restricts_citations(client):
    resp = client.post(
        "/ask",
        json={"question": "How are money mules recruited?", "source": "money_mules.md"},
    )
    body = resp.json()
    assert body["refused"] is False
    assert all(c["source"] == "money_mules.md" for c in body["citations"])


def test_query_log_records_answers_and_refusals(tmp_path):
    log = tmp_path / "queries.jsonl"
    tc = TestClient(_two_doc_app(log_path=log))
    tc.post("/ask", json={"question": "What is an authorized push payment scam?"})
    tc.post("/ask", json={"question": "How do I bake sourdough bread at home?"})
    import json as _json
    records = [_json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["refused"] is False
    assert "app_scams.md" in records[0]["sources"]
    assert records[1]["refused"] is True
    assert records[1]["best_score"] < 0.25


def test_rerank_app_still_answers():
    tc = TestClient(_two_doc_app(log_path="", rerank=True))
    resp = tc.post("/ask", json={"question": "What is an authorized push payment scam?"})
    body = resp.json()
    assert body["refused"] is False
    assert body["citations"][0]["source"] == "app_scams.md"


def test_rrf_app_health_reports_fusion():
    tc = TestClient(_two_doc_app(log_path="", fusion="rrf"))
    body = tc.get("/health").json()
    assert body["fusion"] == "rrf"
