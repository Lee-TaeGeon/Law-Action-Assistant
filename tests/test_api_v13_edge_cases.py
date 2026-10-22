
from fastapi.testclient import TestClient

from backend import main_v13_test as api


def source(law, article, title):
    return {
        "law_name": law,
        "article": article,
        "article_title": title,
        "content": "테스트 법령 원문",
    }


def test_multiple_sources_deduplicated(monkeypatch):
    fraud = source("형법", "제347조", "사기")
    tort = source("민법", "제750조", "불법행위의 내용")
    unrelated = source("민법", "제823조", "혼인취소")

    def fake_chat(*, question, session_id):
        return {
            "category": "형사",
            "answer": (
                "## 확인된 법령 원문\n"
                "- **형법 제347조 (사기)**\n"
                "- **민법 제750조 (불법행위의 내용)**"
            ),
            "sources": [fraud, fraud, tort, unrelated],
            "answer_validation_status": "ok",
            "evidence_selection_mode": "llm_verified",
        }

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/chat",
            json={"question": "법률 질문", "session_id": "case-1"},
        )

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) == 2
    assert [(s["law_name"], s["article"]) for s in sources] == [
        ("형법", "제347조"),
        ("민법", "제750조"),
    ]


def test_session_ids_forwarded(monkeypatch):
    received = []

    def fake_chat(*, question, session_id):
        received.append((question, session_id))
        return {
            "category": "기타",
            "answer": "테스트 답변",
            "sources": [],
            "answer_validation_status": "no_sources",
        }

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    with TestClient(api.app) as client:
        for question, session_id in (
            ("첫 질문", "case-A"),
            ("후속 질문", "case-A"),
            ("다른 상담", "case-B"),
        ):
            response = client.post(
                "/api/chat",
                json={"question": question, "session_id": session_id},
            )
            assert response.status_code == 200

    assert received == [
        ("첫 질문", "case-A"),
        ("후속 질문", "case-A"),
        ("다른 상담", "case-B"),
    ]


def test_blocked_answer_hides_quoted_sources(monkeypatch):
    def fake_chat(*, question, session_id):
        return {
            "category": "형사",
            "answer": "- **형법 제347조 (사기)**",
            "sources": [source("형법", "제347조", "사기")],
            "answer_validation_status": "blocked",
        }

    monkeypatch.setattr(api, "run_legal_chat", fake_chat)

    with TestClient(api.app) as client:
        response = client.post(
            "/api/chat",
            json={"question": "법률 질문", "session_id": "case-2"},
        )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.headers["X-Legal-Validation-Status"] == "blocked"


def test_cors_preflight_for_react():
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
