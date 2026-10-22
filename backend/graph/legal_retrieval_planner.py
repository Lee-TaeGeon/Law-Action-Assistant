
import json

from pydantic import ValidationError

from backend.schemas.legal_case_analysis import LegalCaseAnalysis
from backend.schemas.retrieval_plan import (
    RetrievalPlan,
    validate_request_coverage,
)


class RetrievalPlanningError(ValueError):
    """검색 계획 생성 또는 검증 실패."""


def build_retrieval_plan_prompt(
    analysis: LegalCaseAnalysis,
) -> str:
    requests = [
        {
            "request_index": index,
            "request": request,
        }
        for index, request in enumerate(
            analysis.user_requests,
            start=1,
        )
    ]

    context = {
        "user_stated_facts": analysis.user_stated_facts,
        "user_requests": requests,
        "possible_legal_issues": [
            issue.model_dump()
            for issue in analysis.possible_legal_issues
        ],
        "missing_facts": analysis.missing_facts,
    }

    context_json = json.dumps(
        context,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
당신은 한국 법률 AI 에이전트의 검색 계획 생성기입니다.

역할:
사건 분석 결과를 참고해 사용자 요청마다
필요한 근거 자료의 검색 계획을 생성합니다.

규칙:
1. 모든 user_requests를 빠짐없이 처리하세요.
2. 각 요청의 request_index를 그대로 사용하세요.
3. 요청마다 정확히 하나의 계획 항목을 만드세요.
4. 검색이 필요하면 적절한 검색 질문을 생성하세요.
5. 하나의 요청에 여러 검색 작업을 지정할 수 있습니다.
6. 사용자 진술에 없는 사실을 만들어 내지 마세요.
7. 법령이나 판례의 구체적인 번호를 추측하지 마세요.
8. 검색 결과를 이미 확인한 것처럼 작성하지 마세요.
9. 검색 계획과 실제 법적 판단을 구분하세요.
10. 법률 분야별 고정 규칙 없이 요청의 의미를 분석하세요.
11. 부족한 사실은 clarification_questions에 작성하세요.
12. 반드시 JSON 객체만 출력하세요.

source_type 종류:
- law: 법령 및 조문 검색
- precedent: 판례 검색
- official_guidance: 정부기관 등의 공식 절차 안내 검색

주의:
현재 검색 도구가 모두 연결된 것은 아닙니다.
source_type은 필요한 자료의 종류를 나타낼 뿐,
실제 검색 완료를 의미하지 않습니다.

각 요청에는 검색 작업 또는 추가 확인 질문이
하나 이상 있어야 합니다.

사건 분석 결과:
{context_json}

출력 형식:
{{
  "items": [
    {{
      "request_index": 1,
      "searches": [
        {{
          "source_type": "law",
          "query": "검색할 법적 쟁점"
        }}
      ],
      "clarification_questions": [
        "필요한 추가 사실 확인 질문"
      ]
    }}
  ]
}}

items에는 모든 사용자 요청을 포함하세요.
""".strip()


def generate_retrieval_plan(
    analysis: LegalCaseAnalysis,
    llm,
) -> RetrievalPlan:
    """LLM 검색 계획 생성 후 스키마 및 요청 누락 검증."""

    if not analysis.user_requests:
        raise RetrievalPlanningError(
            "검색 계획을 생성할 사용자 요청이 없습니다."
        )

    prompt = build_retrieval_plan_prompt(analysis)

    try:
        response = llm.invoke(prompt)
        content = response.content

        if not isinstance(content, str):
            raise RetrievalPlanningError(
                "LLM 응답이 문자열 형식이 아닙니다."
            )

        data = json.loads(content)

        plan = RetrievalPlan.model_validate(data)

        validate_request_coverage(
            plan,
            analysis.user_requests,
        )

        return plan

    except (
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise RetrievalPlanningError(
            f"검색 계획 검증 실패: {exc}"
        ) from exc
