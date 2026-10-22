
import json
import sys
from pathlib import Path
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from backend.graph.legal_case_analyzer import analyze_legal_case


ROOT = Path(__file__).resolve().parent


class RequestJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_index: int = Field(ge=1)
    status: Literal["covered", "partial", "missing"]
    reason: str


class CoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[RequestJudgment]


def load_case(case_id: str):
    cases_data = json.loads(
        (ROOT / "legal_cases.json").read_text(encoding="utf-8")
    )
    expectations_data = json.loads(
        (ROOT / "case_analysis_expectations.json").read_text(
            encoding="utf-8"
        )
    )

    case = next(
        c for c in cases_data["cases"] if c["id"] == case_id
    )
    expected = next(
        c for c in expectations_data["cases"] if c["id"] == case_id
    )

    return case, expected


def evaluate_request_coverage(expected_requests, actual_requests, llm):
    prompt = f"""
당신은 법률 사건 분석기의 요청 추출 품질 평가자입니다.
법적 정답을 판단하지 말고 사용자 요청이 보존됐는지만 평가하세요.

평가 기준:
1. 문자열 일치가 아니라 의미를 비교합니다.
2. 요청의 핵심 목적이 모두 보존되면 covered입니다.
3. 핵심 목적의 일부만 보존되면 partial입니다.
4. 해당 요청을 찾을 수 없으면 missing입니다.
5. 실제 분석에 없는 내용을 추측해 보완하지 마세요.
6. 모든 expected_index를 빠짐없이 한 번씩 평가하세요.
7. JSON 객체만 출력하세요.

평가할 요청:
{json.dumps(expected_requests, ensure_ascii=False)}

분석기가 추출한 요청:
{json.dumps(actual_requests, ensure_ascii=False)}

출력 형식:
{{
  "judgments": [
    {{
      "expected_index": 1,
      "status": "covered",
      "reason": "판정 근거"
    }}
  ]
}}

expected_index는 평가할 요청 목록의 1부터 시작하는 번호입니다.
"""

    response = llm.invoke(prompt)
    report = CoverageReport.model_validate(
        json.loads(response.content)
    )

    expected_indices = list(range(1, len(expected_requests) + 1))
    actual_indices = sorted(
        item.expected_index for item in report.judgments
    )

    if actual_indices != expected_indices:
        raise ValueError(
            "평가 결과에 누락되거나 중복된 요청 번호가 있습니다."
        )

    return report


def main():
    case_id = sys.argv[1] if len(sys.argv) > 1 else "LAW-001"
    case, expectation = load_case(case_id)

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
        format="json",
    )

    analysis = analyze_legal_case(case["question"], llm)

    report = evaluate_request_coverage(
        expectation["required_user_requests"],
        analysis.user_requests,
        llm,
    )

    print(f"\n=== {case_id} 요청 누락 평가 ===")
    print("분석기 추출 요청:")
    print(
        json.dumps(
            analysis.user_requests,
            ensure_ascii=False,
            indent=2,
        )
    )

    for judgment in report.judgments:
        expected_request = expectation["required_user_requests"][
            judgment.expected_index - 1
        ]

        print(f"\n기준: {expected_request}")
        print(f"판정: {judgment.status}")
        print(f"이유: {judgment.reason}")

    print(
        "\n참고: LLM 자동 평가 결과이며 법률 전문가 검토를 "
        "대체하지 않습니다."
    )


if __name__ == "__main__":
    main()
