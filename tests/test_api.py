import json

import pytest
from fastapi.testclient import TestClient

from agritrial_llm.config import ServiceSettings, get_settings
from agritrial_llm.main import app


@pytest.fixture
def client(settings: ServiceSettings):
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def chat_body(dataset, question: str, **options) -> dict:
    return {"question": question, "dataset": dataset.model_dump(mode="json"), "options": options}


def test_health_reports_ollama_status(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama"]["reachable"] is False
    assert body["defaults"]["model"] == "qwen3.6"


def test_chat_endpoint(client, dataset):
    response = client.post("/v1/agent/chat", json=chat_body(dataset, "Which trials have conflicting data?"))
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "offline"
    assert "T03" in body["answer"]
    assert body["trace"][0]["message"].startswith("[Intent]")


def test_stream_endpoint_sends_server_sent_events(client, dataset):
    with client.stream("POST", "/v1/agent/stream", json=chat_body(dataset, "Compare T01 and T04")) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[len("data: "):]) for line in response.iter_lines() if line.startswith("data: ")]
    assert [event["type"] for event in events[-2:]] == ["answer", "done"]
    assert "T04" in events[-2]["message"]


def test_chat_without_fallback_returns_503(client, dataset):
    response = client.post("/v1/agent/chat", json=chat_body(dataset, "List trials", offline_fallback=False))
    assert response.status_code == 503


def test_api_key_is_enforced(dataset):
    app.dependency_overrides[get_settings] = lambda: ServiceSettings(
        _env_file=None, ollama_base_url="http://127.0.0.1:9", llm_service_api_key="secret"
    )
    try:
        client = TestClient(app)
        body = chat_body(dataset, "List trials")
        assert client.post("/v1/agent/chat", json=body).status_code == 401
        assert client.post("/v1/agent/chat", json=body, headers={"X-API-Key": "secret"}).status_code == 200
        assert client.get("/health").status_code == 200  # health stays open for monitoring
    finally:
        app.dependency_overrides.clear()


def test_graph_endpoint_shows_both_loops(client):
    body = client.get("/v1/agent/graph").json()
    assert "tools" in body["agent"] and "verify" in body["agent"] and "revise" in body["agent"]
    assert "validate" in body["extraction"]


def test_llm_test_endpoint_reports_unreachable_ollama(client):
    body = client.post("/v1/llm/test", json={}).json()
    assert body["ok"] is False
    assert "not reachable" in body["error"]


def test_request_validation(client, dataset):
    assert client.post("/v1/agent/chat", json=chat_body(dataset, "")).status_code == 422
