import pytest
from fastapi.testclient import TestClient

from docqa.api import create_app
from docqa.llm import ExtractiveGenerator


@pytest.fixture()
def client(index):
    with TestClient(create_app(index=index, generator=ExtractiveGenerator())) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["chunks"] > 0


def test_ask_returns_grounded_answer_with_sources(client):
    r = client.post("/ask", json={"question": "How do I return a 404 when an item is not found?", "k": 3})
    body = r.json()
    assert r.status_code == 200
    assert body["grounded"] and body["sources"][0]["doc"] == "errors.md"
    assert len(body["sources"]) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},  # too short
        {"question": "valid question", "k": 0},  # k out of range
        {"question": "valid question", "mode": "magic"},
    ],
)
def test_ask_validates_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_metrics_track_requests(client):
    for _ in range(3):
        client.post("/ask", json={"question": "CORS for my frontend"})
    m = client.get("/metrics").json()
    assert m["requests"] == 3 and m["latency_ms_p95"] >= m["latency_ms_p50"] >= 0
    assert m["grounded_rate"] == 1.0


def test_routes_also_served_under_api_prefix(client):
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/ask", json={"question": "CORS for my frontend"}).json()["sources"]
