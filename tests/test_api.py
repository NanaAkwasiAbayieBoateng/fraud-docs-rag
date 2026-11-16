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
    app = create_app(client=fake, index=index, refusal_threshold=0.25)
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
    app = create_app(client=fake, index=HybridIndex(), refusal_threshold=0.25)
    resp = TestClient(app).post("/ask", json={"question": "What is a money mule?"})
    assert resp.json()["refused"] is True
