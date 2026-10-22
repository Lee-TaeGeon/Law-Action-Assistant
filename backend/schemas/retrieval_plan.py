
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


PlanText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=500,
    ),
]


class SearchInstruction(BaseModel):
    """검색할 근거의 종류와 검색 질문."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal[
        "law",
        "precedent",
        "official_guidance",
    ]
    query: PlanText


class RequestPlanItem(BaseModel):
    """하나의 사용자 요청을 처리하기 위한 계획."""

    model_config = ConfigDict(extra="forbid")

    request_index: int = Field(ge=1)
    searches: list[SearchInstruction] = Field(
        default_factory=list,
        max_length=8,
    )
    clarification_questions: list[PlanText] = Field(
        default_factory=list,
        max_length=5,
    )

    @model_validator(mode="after")
    def require_action(self):
        if not self.searches and not self.clarification_questions:
            raise ValueError(
                "검색 또는 추가 확인 질문이 하나 이상 필요합니다."
            )
        return self


class RetrievalPlan(BaseModel):
    """사용자 요청별 검색 및 추가 확인 계획."""

    model_config = ConfigDict(extra="forbid")

    items: list[RequestPlanItem] = Field(
        min_length=1,
        max_length=10,
    )


def validate_request_coverage(
    plan: RetrievalPlan,
    user_requests: list[str],
) -> None:
    """모든 사용자 요청이 정확히 하나의 계획에 연결되는지 검사."""

    if not user_requests:
        raise ValueError("평가할 사용자 요청이 없습니다.")

    expected = list(range(1, len(user_requests) + 1))
    actual = sorted(item.request_index for item in plan.items)

    if actual != expected:
        raise ValueError(
            f"요청별 검색 계획이 불완전합니다. "
            f"expected={expected}, actual={actual}"
        )
