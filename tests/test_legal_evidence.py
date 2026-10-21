import json
from types import SimpleNamespace

import pytest

from backend.graph import legal_graph_v8 as v8
from backend.graph import legal_graph_v9 as v9


@pytest.fixture
def source():
    return {
        "law_name": "민법",
        "article": "제394조",
        "article_title": "손해배상의 방법",
        "content": (
            "<민법>\n"
            "제394조 (손해배상의 방법)\n"
            "다른 의사표시가 없으면 손해는 금전으로 배상한다."
        ),
    }


@pytest.fixture
def quote():
    return "다른 의사표시가 없으면 손해는 금전으로 배상한다."


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(
            content=next(self.outputs)
        )


def test_valid_quote_accepted(source, quote):
    payload = {
        "findings": [
            {"source_id": "S1", "evidence": quote}
        ]
    }

    result = v8.validate_evidence_draft(
        payload, [source]
    )

    assert len(result) == 1
    assert result[0]["evidence"] == quote
    assert result[0]["source"] == source


def test_fabricated_quote_rejected(source):
    payload = {
        "findings": [
            {
                "source_id": "S1",
                "evidence": (
                    "모든 손해배상은 예외 없이 "
                    "즉시 현금으로 지급해야 한다."
                ),
            }
        ]
    }

    with pytest.raises(v8.EvidenceError):
        v8.validate_evidence_draft(
            payload, [source]
        )


def test_article_header_not_accepted(source):
    payload = {
        "findings": [
            {
                "source_id": "S1",
                "evidence": "제394조 (손해배상의 방법)",
            }
        ]
    }

    with pytest.raises(v8.EvidenceError):
        v8.validate_evidence_draft(
            payload, [source]
        )


def test_extra_json_field_rejected():
    response = json.dumps({
        "findings": [],
        "explanation": "검증되지 않은 설명",
    }, ensure_ascii=False)

    with pytest.raises(v8.EvidenceError):
        v8.parse_evidence_draft(response)


def test_valid_llm_quote_preserved(source, quote):
    good = json.dumps({
        "findings": [
            {"source_id": "S1", "evidence": quote}
        ]
    }, ensure_ascii=False)

    model = FakeLLM([good])

    result = v9.answer_from_sources(
        "손해배상 방법을 알려주세요.",
        [source],
        model,
    )

    assert result["answer_validation_status"] == "ok"
    assert result["evidence_selection_mode"] == "llm_verified"
    assert quote in result["answer"]
    assert model.calls == 1


def test_invalid_llm_quote_fallback(source, quote):
    bad = json.dumps({
        "findings": [
            {"source_id": "S1", "evidence": "금전"}
        ]
    }, ensure_ascii=False)

    model = FakeLLM([bad, bad])

    result = v9.answer_from_sources(
        "계약에서 손해배상은 어떻게 하나요?",
        [source],
        model,
    )

    assert result["answer_validation_status"] == "ok"
    assert (
        result["evidence_selection_mode"]
        == "deterministic_fallback"
    )
    assert quote in result["answer"]
    assert model.calls == 2


def test_fallback_blocks_without_valid_text():
    broken = {
        "law_name": "테스트법",
        "article": "제0조",
        "content": "짧음",
    }

    bad = json.dumps({
        "findings": [
            {"source_id": "S1", "evidence": "금전"}
        ]
    }, ensure_ascii=False)

    model = FakeLLM([bad, bad])

    result = v9.answer_from_sources(
        "테스트 질문",
        [broken],
        model,
    )

    assert result["answer_validation_status"] == "blocked"
    assert result["evidence_selection_mode"] == "none"
    assert model.calls == 2


def test_no_sources_handled():
    model = FakeLLM([])

    result = v9.answer_from_sources(
        "관련 법령을 알려주세요.",
        [],
        model,
    )

    assert result["answer_validation_status"] == "no_sources"
    assert result["evidence_selection_mode"] == "none"
    assert model.calls == 0