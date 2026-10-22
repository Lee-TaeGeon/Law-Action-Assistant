
"""범용 법률 사건 분석 스키마 단위 테스트.

LLM, SQLite, ChromaDB를 호출하지 않는다.
데이터 구조와 입력값 검증만 확인한다.
"""

import pytest
from pydantic import ValidationError

from backend.schemas.legal_case_analysis import (
    LegalCaseAnalysis,
)


def valid_analysis():
    """테스트용 가상 사건 분석 데이터."""
    return {
        "user_stated_facts": [
            "사용자가 대금을 지급했다고 진술함",
            "상대방이 물건을 보내지 않았다고 진술함",
        ],
        "user_requests": [
            "형사책임 성립 가능성 분석",
            "피해금 반환 방법 안내",
        ],
        "possible_legal_issues": [
            {
                "issue": "형사상 사기 성립 여부",
                "search_query": "형법 사기죄 성립 요건",
            },
            {
                "issue": "민사상 금전 반환 문제",
                "search_query": "계약 불이행 해제 및 반환청구",
            },
        ],
        "missing_facts": [
            "거래 당시 상대방의 설명",
            "대금 지급 날짜",
        ],
    }


def test_valid_case_analysis():
    """정상적인 사건 분석 데이터를 수용한다."""
    result = LegalCaseAnalysis.model_validate(
        valid_analysis()
    )

    assert len(result.user_requests) == 2
    assert len(result.possible_legal_issues) == 2
    assert (
        result.possible_legal_issues[0].search_query
        == "형법 사기죄 성립 요건"
    )


def test_unknown_field_rejected():
    """정의하지 않은 필드를 차단한다."""
    data = valid_analysis()
    data["legal_conclusion"] = "사기죄 확정"

    with pytest.raises(ValidationError):
        LegalCaseAnalysis.model_validate(data)


def test_empty_issue_rejected():
    """비어 있는 법적 쟁점 문장을 차단한다."""
    data = valid_analysis()
    data["possible_legal_issues"][0]["issue"] = "   "

    with pytest.raises(ValidationError):
        LegalCaseAnalysis.model_validate(data)


def test_too_many_requests_rejected():
    """설정된 최대 요청 항목 수를 초과하면 차단한다."""
    data = valid_analysis()
    data["user_requests"] = [
        f"요청사항 {i}" for i in range(11)
    ]

    with pytest.raises(ValidationError):
        LegalCaseAnalysis.model_validate(data)


def test_missing_required_field_rejected():
    """필수 필드가 누락되면 차단한다."""
    data = valid_analysis()
    del data["user_stated_facts"]

    with pytest.raises(ValidationError):
        LegalCaseAnalysis.model_validate(data)
