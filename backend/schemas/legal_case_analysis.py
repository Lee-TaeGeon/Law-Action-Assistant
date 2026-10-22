
"""범용 법률 사건 분석 결과 데이터 스키마.

역할:
- LLM이 분석한 사용자 진술, 요청사항, 법적 쟁점을 구조화한다.
- 검색에 사용할 쟁점별 검색 질문을 저장한다.
- 출력 형식과 입력 크기를 검증한다.

주의:
- 사용자 진술은 검증된 사실이 아니다.
- 가능한 법적 쟁점은 확정된 법률 판단이 아니다.
- 이 스키마는 법률적 정확성을 보증하지 않는다.
- 기존 LangGraph v13 또는 법령 DB를 변경하지 않는다.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)


# 빈 문자열이나 지나치게 긴 텍스트를 제한한다.
FactText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1000,
    ),
]

IssueText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]

SearchText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]


class LegalIssue(BaseModel):
    """LLM이 식별한 잠재적인 법적 쟁점."""

    model_config = ConfigDict(extra="forbid")

    issue: IssueText = Field(
        description="검토가 필요한 법적 쟁점"
    )

    search_query: SearchText = Field(
        description="해당 쟁점의 법령 검색에 사용할 질문"
    )


class LegalCaseAnalysis(BaseModel):
    """사용자 법률 질문의 구조화된 분석 결과."""

    model_config = ConfigDict(extra="forbid")

    user_stated_facts: list[FactText] = Field(
        ...,
        max_length=20,
        description=(
            "사용자가 직접 진술한 사실. "
            "객관적으로 검증된 사실로 취급하지 않는다."
        ),
    )

    user_requests: list[IssueText] = Field(
        ...,
        max_length=10,
        description="사용자가 답변받기를 원하는 요청사항",
    )

    possible_legal_issues: list[LegalIssue] = Field(
        ...,
        max_length=10,
        description=(
            "검토할 가능성이 있는 법적 쟁점 목록. "
            "법적 판단이 확정되었다는 의미가 아니다."
        ),
    )

    missing_facts: list[FactText] = Field(
        ...,
        max_length=15,
        description=(
            "법적 쟁점을 분석하기 위해 "
            "추가 확인이 필요한 사실관계"
        ),
    )
