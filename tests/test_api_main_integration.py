
"""FastAPI main.py v13 integration contract tests."""

from fastapi.testclient import TestClient

from backend import main as api
from backend.graph.legal_graph_v13 import run_legal_chat as v13_chat


def test_main_uses_v13_graph():
    assert api.run_legal_chat is v13_chat


def test_main_verified_sources_and_headers(monkeypatch):
    def fake_chat(*, question, session_id):
        assert question == "중고거래 법률 질문"
        assert session_id == "integration-case-001"

        return {
            "category": "형사",
            "answer": (
                "## 확인된 법령 원문\n"
                "- **형법 제347조 (사기)**\n"
                "  - 법문 원문: 테스트 인용"
            ),
            "sources": [
                {
                    "law_name": "형법",
                    "article": "제347조",
                    "article_title": "사기",
                    "content": "테스트 형법 원문",
                },
                {
                    "law_name": "민법",
                    "article": "제750조",
                    "article_title": "불법행위의 내용",
                    "content": "테스트 민법 원문",
                },
            ],
            "answer_validation_status": "ok",
            "evidence_selection_mode": "llm_verified",
        }

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/chat",
            json={
                "question": "중고거래 법률 질문",
                "session_id": "integration-case-001",
            },
        )

    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1
    assert response.json()["sources"][0]["law_name"] == "형법"
    assert response.headers["X-Legal-Validation-Status"] == "ok"
    assert response.headers["X-Legal-Evidence-Mode"] == "llm_verified"


def test_main_blocks_unverified_sources(monkeypatch):
    def fake_chat(*, question, session_id):
        return {
            "category": "형사",
            "answer": "- **형법 제347조 (사기)**",
            "sources": [
                {
                    "law_name": "형법",
                    "article": "제347조",
                    "article_title": "사기",
                    "content": "테스트 법문",
                }
            ],
            "answer_validation_status": "blocked",
        }

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/chat",
            json={
                "question": "법률 질문",
                "session_id": "integration-case-002",
            },
        )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.headers["X-Legal-Validation-Status"] == "blocked"


def test_main_react_cors_preflight():
    with TestClient(api.app) as client:
        response = client.options(
            "/api/chat",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://127.0.0.1:5173"
    )
