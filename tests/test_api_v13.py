import pytest
from fastapi.testclient import TestClient

from backend import main_v13_test as api


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["mode"] == "v13-test"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["mode"] == "v13-test"
    assert data["law_search_documents"] == 205819
    assert data["vector_documents"] == 205819


def test_chat_contract_and_sources(client, monkeypatch):
    received = {}

    def fake_chat(*, question, session_id):
        received["question"] = question
        received["session_id"] = session_id

        return {
            "category": "형사",
            "answer": (
                "## 확인된 법령 원문\n"
                "- **형법 제347조 (사기)**\n"
                "  - 법문 원문: 테스트 인용문"
            ),
            "sources": [
                {
                    "law_name": "형법",
                    "article": "제347조",
                    "article_title": "사기",
                    "content": "테스트용 형법 원문",
                },
                {
                    "law_name": "민법",
                    "article": "제750조",
                    "article_title": "불법행위의 내용",
                    "content": "테스트용 민법 원문",
                },
            ],
            "answer_validation_status": "ok",
            "evidence_selection_mode": "llm_verified",
        }

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={
            "question": "중고거래 사기 질문",
            "session_id": "test-session-001",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "형사"
    assert "형법 제347조" in data["answer"]

    # 실제 답변에 인용된 형법만 출처 카드로 표시
    assert len(data["sources"]) == 1
    assert data["sources"][0]["law_name"] == "형법"
    assert data["sources"][0]["article"] == "제347조"

    assert received["session_id"] == "test-session-001"
    assert received["question"] == "중고거래 사기 질문"

    assert (
        response.headers["X-Legal-Validation-Status"]
        == "ok"
    )
    assert (
        response.headers["X-Legal-Evidence-Mode"]
        == "llm_verified"
    )


def test_unverified_sources_hidden(client, monkeypatch):
    def fake_chat(*, question, session_id):
        return {
            "category": "형사",
            "answer": "검증 실패로 답변을 차단했습니다.",
            "sources": [
                {
                    "law_name": "형법",
                    "article": "제347조",
                    "content": "테스트 법문",
                }
            ],
            "answer_validation_status": "blocked",
        }

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={
            "question": "법률 질문",
            "session_id": "test-002",
        },
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert (
        response.headers["X-Legal-Validation-Status"]
        == "blocked"
    )


def test_invalid_request(client):
    response = client.post(
        "/api/chat",
        json={"session_id": "test-003"},
    )

    assert response.status_code == 422


def test_rate_limit(client, monkeypatch):
    def fake_chat(*, question, session_id):
        raise RuntimeError("rate_limit_exceeded")

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={
            "question": "법률 질문",
            "session_id": "test-004",
        },
    )

    assert response.status_code == 429


def test_internal_error(client, monkeypatch):
    def fake_chat(*, question, session_id):
        raise RuntimeError("Unexpected internal error")

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={
            "question": "법률 질문",
            "session_id": "test-005",
        },
    )

    assert response.status_code == 500


def test_cors(client):
    response = client.get(
        "/",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:5173"
    )