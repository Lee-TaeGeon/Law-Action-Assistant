
"""범용 법률 사건·쟁점 분석기 단위 테스트.

실제 Ollama, SQLite, ChromaDB는 호출하지 않는다.
FakeLLM을 이용해 출력 구조와 실패 처리를 검증한다.
"""

import json
from types import SimpleNamespace

import pytest

from backend.graph.legal_case_analyzer import (
    CaseAnalysisError,
    analyze_legal_case,
    build_case_analysis_prompt,
)


class FakeLLM:
    """테스트용 LLM."""

    def __init__(self, content):
        self.content = content
        self.call_count = 0
        self.last_prompt = None

    def invoke(self, prompt):
        self.call_count += 1
        self.last_prompt = prompt
        return SimpleNamespace(content=self.content)


def valid_payload():
    """가상의 민사·형사 복합 사건 분석 결과."""
    return {
        "user_stated_facts": [
            "사용자가 판매자에게 대금을 지급했다고 진술함",
            "판매자가 물건을 보내지 않았다고 진술함",
        ],
        "user_requests": [
            "형사상 사기 성립 가능성 분석",
            "민사상 피해금 반환 방법 안내",
        ],
        "possible_legal_issues": [
            {
                "issue": "형사상 사기 성립 여부",
                "search_query": "형법 사기죄 성립 요건",
            },
            {
                "issue": "민사상 계약 불이행 및 반환",
                "search_query": "계약 불이행 해제 원상회복",
            },
        ],
        "missing_facts": [
            "거래 당시 판매자의 설명",
            "대금 지급 및 배송 약속 날짜",
        ],
    }


def test_valid_analysis():
    """정상적인 LLM 분석 결과를 반환한다."""

    llm = FakeLLM(
        json.dumps(valid_payload(), ensure_ascii=False)
    )

    question = (
        "중고거래에서 돈을 보냈는데 물건을 받지 못했습니다. "
        "사기죄가 성립하나요? 돈도 돌려받고 싶습니다."
    )

    result = analyze_legal_case(question, llm)

    assert llm.call_count == 1
    assert question in llm.last_prompt

    assert len(result.user_requests) == 2
    assert len(result.possible_legal_issues) == 2
    assert len(result.missing_facts) == 2


def test_invalid_json_rejected():
    """잘못된 JSON은 차단한다."""

    llm = FakeLLM("올바르지 않은 JSON 응답")

    with pytest.raises(CaseAnalysisError):
        analyze_legal_case("법률 질문입니다.", llm)


def test_unknown_field_rejected():
    """LLM이 임의로 추가한 필드는 차단한다."""

    payload = valid_payload()
    payload["confirmed_legal_conclusion"] = "사기죄 확정"

    llm = FakeLLM(
        json.dumps(payload, ensure_ascii=False)
    )

    with pytest.raises(CaseAnalysisError):
        analyze_legal_case("법률 질문입니다.", llm)


def test_missing_requests_rejected():
    """사용자 요청사항을 식별하지 못하면 차단한다."""

    payload = valid_payload()
    payload["user_requests"] = []

    llm = FakeLLM(
        json.dumps(payload, ensure_ascii=False)
    )

    with pytest.raises(CaseAnalysisError):
        analyze_legal_case("법률 질문입니다.", llm)


def test_llm_failure_handled():
    """LLM 호출 실패 시 명확한 예외를 발생시킨다."""

    class FailingLLM:
        def invoke(self, prompt):
            raise RuntimeError("모델 연결 실패")

    with pytest.raises(
        CaseAnalysisError,
        match="모델 호출에 실패",
    ):
        analyze_legal_case(
            "법률 질문입니다.",
            FailingLLM(),
        )


def test_empty_question_rejected_before_llm_call():
    """빈 질문은 LLM 호출 전에 차단한다."""

    llm = FakeLLM(
        json.dumps(valid_payload(), ensure_ascii=False)
    )

    with pytest.raises(ValueError):
        analyze_legal_case("   ", llm)

    assert llm.call_count == 0
