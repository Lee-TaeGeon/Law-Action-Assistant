from types import SimpleNamespace

import pytest

from backend.graph import legal_graph_v4 as v4
from backend.graph import legal_graph_v13 as v13


def make_source(law, article, title):
    return {
        "law_name": law,
        "article": article,
        "article_title": title,
        "section_scope": "",
        "content": "테스트용 법령 본문",
    }


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=next(self.outputs))


def mock_llm(monkeypatch, outputs):
    model = FakeLLM(outputs)
    monkeypatch.setattr(
        v4.original,
        "get_fast_llm",
        lambda: model,
    )
    return model


def make_state(question, sources):
    return {
        "incoming_question": question,
        "question": question,
        "category": "형사",
        "candidate_sources": sources,
    }


def test_irrelevant_article_filtered(monkeypatch):
    fraud = make_source("형법", "제347조", "사기")
    family = make_source(
        "민법", "제823조",
        "사기, 강박으로 인한 혼인취소청구권의 소멸",
    )
    tort = make_source(
        "민법", "제750조", "불법행위의 내용"
    )

    model = mock_llm(monkeypatch, ["C00,C01"])

    state = make_state(
        "중고거래 사기와 민사상 손해배상",
        [fraud, family, tort],
    )

    result = v13.reranker_node(state)

    assert result["reranker_status"] == "ok"
    assert result["sources"] == [fraud, tort]
    assert model.calls == 1


def test_invalid_response_retry(monkeypatch):
    fraud = make_source("형법", "제347조", "사기")

    model = mock_llm(
        monkeypatch,
        ["invalid response", "C00"],
    )

    result = v13.reranker_node(
        make_state("중고거래 사기", [fraud])
    )

    assert result["reranker_status"] == "ok"
    assert result["sources"] == [fraud]
    assert model.calls == 2


def test_no_relevant_candidates(monkeypatch):
    fraud = make_source("형법", "제347조", "사기")

    model = mock_llm(monkeypatch, ["없음"])

    result = v13.reranker_node(
        make_state("중고거래 사기", [fraud])
    )

    assert result["reranker_status"] == "none"
    assert result["sources"] == []
    assert model.calls == 1


def test_retry_failure_blocks(monkeypatch):
    fraud = make_source("형법", "제347조", "사기")

    model = mock_llm(
        monkeypatch,
        ["invalid", "still invalid"],
    )

    result = v13.reranker_node(
        make_state("중고거래 사기", [fraud])
    )

    assert result["reranker_status"] == "format_error"
    assert result["sources"] == []
    assert model.calls == 2


def test_all_filtered_skips_llm(monkeypatch):
    family = make_source(
        "민법", "제823조",
        "사기, 강박으로 인한 혼인취소청구권의 소멸",
    )

    def unexpected_llm():
        pytest.fail("LLM should not be called")

    monkeypatch.setattr(
        v4.original,
        "get_fast_llm",
        unexpected_llm,
    )

    result = v13.reranker_node(
        make_state("중고거래 사기", [family])
    )

    assert result == {
        "sources": [],
        "reranker_status": "none",
    }


def test_explicit_article_preserved(monkeypatch):
    commerce = make_source(
        "전자상거래 등에서의 소비자보호에 관한 법률",
        "제6조",
        "거래기록의 보존 등",
    )

    model = mock_llm(monkeypatch, ["C00"])

    result = v13.reranker_node(
        make_state(
            "전자상거래법 제6조를 설명해 주세요",
            [commerce],
        )
    )

    assert result["reranker_status"] == "ok"
    assert result["sources"] == [commerce]
    assert model.calls == 1